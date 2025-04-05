# Function for extracting file metadata
def extract_code_metadata(code):
    try:
        tree = ast.parse(code)
        functions = [node.name for node in ast.walk(
            tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(
            tree) if isinstance(node, ast.ClassDef)]
        return functions, classes
    except Exception:
        return [], []


# Function to recursively get all code files with specified extension in a folder
def get_code_files(data_path, extensions):
    code_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(tuple(extensions)):
                code_files.append(os.path.join(root, file))
    return code_files


# Function to process a single file
def process_file(file_path, data_path, store_full_content=True):
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        relative_path = str(Path(file_path).relative_to(data_path))
        file_extension = Path(file_path).suffix.lower()
        functions, classes = extract_code_metadata(content)

        metadata = {
            "source": relative_path,
            "extension": file_extension,
            "classes": ", ".join(classes) if classes else "None",
            "functions": ", ".join(functions) if functions else "None",
            "full_content": content if store_full_content else ""
        }
        return Document(
            page_content=content,
            metadata=metadata
        )
    except Exception as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Processing {Path(file_path).name}: {str(e)}")
        return None


# Function to load files and process them in parallel
def load_files_as_documents(data_path, extensions, llm, llm_summary=False):
    code_files = get_code_files(data_path, extensions)
    docs = []

    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_file, file, data_path, llm_summary, llm): file
            for file in code_files
        }

        for future in tqdm(
                as_completed(future_to_file),
                total=len(code_files),
                desc=f"{Fore.BLUE}Processing files{Style.RESET_ALL}",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            doc = future.result()
            if doc:
                docs.append(doc)

    return docs

def generate_chunk_context(llm, full_content: str, chunk_content: str) -> str:
    prompt = f"""Analyze this code file and explain how the following excerpt fits into the whole:
            Full File Content:
            {full_content[:8000]}  # Limit to first 10k chars to avoid token limits
            
            Code Excerpt:
            {chunk_content}
            
            Provide a concise 2-3 sentence explanation of:
            1. How this excerpt relates to the overall file structure
            2. Its functional purpose within the file
            3. Key classes/functions it interacts with
            
            Answer in a technical but concise manner, focusing on code relationships."""
    
    llm = get_llm(llm, max_tokens=600)
    response = llm.invoke(prompt)
    return response.content.strip()

def contextualize_chunks(split_docs: List[Document], llm, parallel_threads: int = 4) -> List[Document]:
    """Add document-level context to each chunk"""
    contextualized_docs = []
    
    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        futures = []
        for doc in split_docs:
            full_content = doc.page_content
            if not full_content:
                continue
                
            futures.append(
                executor.submit(
                    _process_single_contextual_chunk,
                    doc,
                    full_content,
                    llm
                )
            )
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                         desc=f"{Fore.BLUE}Contextualizing chunks{Style.RESET_ALL}"):
            contextualized_docs.append(future.result())
    
    return [doc for doc in contextualized_docs if doc is not None]

def _process_single_contextual_chunk(doc: Document, chunk: str, llm) -> Optional[Document]:
    """Process individual chunk with context"""
    try:
        context = generate_chunk_context(
            llm,
            doc.metadata['full_content'],
            doc.page_content,
        )
        
        # Create enhanced content with context
        enhanced_content = (
            f"CONTEXTUAL ANALYSIS:\n{context}\n\n"
            f"CODE SUMMARY:\n{doc.page_content}\n\n"
            f"CODE CHUNK:\n{doc.page_content}\n\n"
        )

        doc.metadata['full_content'] = ''

        # Return new document with enhanced content
        return Document(
            page_content=enhanced_content,
            metadata=doc.metadata
        )
    except Exception as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Contextualizing chunk: {str(e)}")
        return None


# Function for creating the vector store
def create_vector_store(
    data_path,
    extensions,
    persistent_directory,
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="default",
    llm_summary=False,
    llm=None,
    use_contextual_chunks=True,
    parallel_context_threads=3
):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== Creating Contextual Vector Store ==={Style.RESET_ALL}")
    
    # Load and split documents
    docs = load_files_as_documents(data_path, extensions, llm, llm_summary)
    if not docs:
        print(f"{Fore.RED}No documents found. Exiting.{Style.RESET_ALL}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"{Fore.GREEN}Base chunks created:{Style.RESET_ALL} {len(split_docs)}")

    # Add contextual layer
    if use_contextual_chunks:
        split_docs = contextualize_chunks(split_docs, llm, parallel_context_threads)
        print(f"{Fore.GREEN}Contextualized chunks:{Style.RESET_ALL} {len(split_docs)}")

    # Create vector store
    embeddings = get_embeddings_function(embedding_model)
    db = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persistent_directory,
        collection_metadata={
            "hnsw:space": "cosine",
            "contextual_embedding": str(use_contextual_chunks)
        },
    )
    
    print(f"{Fore.GREEN}Vector store created at:{Style.RESET_ALL} {persistent_directory}")
    return db