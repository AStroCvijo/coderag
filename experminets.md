
# **Evaluation of Retrieval-Augmented Generation (RAG) for Code Search**

  

## **1. Baseline Performance (Regular RAG with No Modifications)**

The initial setup involved splitting the code into chunks and embedding them using **text-embedding-small**. Using similarity retrieval, this approach achieved the following:

-  **Best-performing setup:**  **Chunk size 700, chunk overlap 100**

-  **Recall@10:**  **63.7%**

  

✅ **Decent performance, but significant room for improvement.**

  

---

  

## **2. Using LLM Summaries for Embeddings**

Since queries are in **natural language**, I experimented with **summarizing the code using an LLM** before embedding. While this made **vector store creation much slower**, it improved retrieval accuracy:

-  **Chunk size 700, overlap 100 → Recall@10: 67.6%**

  

✅ **Better than the baseline, but still far from optimal.**

  

---

  

## **3. Experimenting with Query Expansion (Did Not Work Well)**

Next, I attempted to **expand the queries** using different prompt templates. However, this approach had **several drawbacks**:

-  **Inference time increased significantly.**

-  **Accuracy dropped across all chunk sizes and overlaps.**

-  **Best attempt (shortened expanded queries) reached only Recall@10: 64.3%, still worse than the original queries.**

  

❌ **Conclusion: Query expansion is not worth it for this task.**

  

---

  

## **4. Improving Summarization & Using a Bigger Embedding Model**

To further enhance retrieval, I:

- Used **a better summarization prompt with more metadata**.

- Switched from **text-embedding-small** to **text-embedding-large** (slightly more expensive but potentially more effective).

  

Results:

-  **Chunk size 1100, overlap 100 → Recall@10: 69.5%** (Best for this setting)

-  **Chunk size 700, overlap 100 → Recall@10: 70.2%** (Best overall)

  

✅ **This was the most effective setup, achieving the highest retrieval accuracy at this stage.**

  

---

  

## **5. Further Enhancements & Best Achieved Performance**

After noticing missing file extensions in the dataset, I included them, which improved accuracy. I then experimented with **adding code snippets alongside the summaries**, leading to a major improvement:

-  **Chunk size 1200, overlap 200 → Recall@10: 80.9%**

  

To push accuracy even further, I enhanced metadata by including **a list of functions and classes present in each file**, which resulted in:

-  **Recall@10: 84.6%**

  

Finally, I found an error in the querying process where files were first sorted by relevance before ensuring uniqueness in the top 10, potentially causing crucial losses. Fixing this boosted accuracy to:

-  **Recall@10: 86.0%** (Best achieved performance)

-  **Recall@40: 90.3%** (Suggesting a re-ranking system could further improve results)

  

✅ **Best RAG system so far, with a significant improvement over the baseline.**

  

---

  

## **6. Experimenting with an LLM-Based Re-Ranker**

I also experimented with an **LLM-based re-ranker** to further refine the retrieval results. However, this approach had **mixed outcomes**:

-  **Inference time increased significantly**, making the system less practical for real-time applications.

-  **Accuracy dropped slightly**, likely due to the re-ranker introducing noise or misaligning with the retrieval objectives.

  

❌ **Conclusion: The LLM-based re-ranker was not effective for this task and is not recommended.**

  

---

  

## **7. Experiments with Other Languages**

During the evaluation, I noticed that some files in the dataset were in **Russian**. This led to an interesting observation:

- When queries were translated into Russian, the RAG system was able to retrieve the **exact Russian README file** that it previously missed. However, in these cases, it failed to retrieve the corresponding **English README file**.

- The overall accuracy of translating queries to Russian was **83.3%**, which is slightly worse than the accuracy achieved with English prompts (**86.0%**).

  

✅ **Key Insights:**
- The RAG system can handle multilingual content effectively, but there is a trade-off in accuracy when translating queries.
- For multilingual datasets, a hybrid approach (e.g., querying in both languages and merging results) might be worth exploring to improve recall without sacrificing accuracy.

  

---

  

## **Summary of Findings**

| **Method** | **Best Chunk Size** | **Best Overlap** | **Recall@10 (%)** | **Notes** |
|------------|-----------------|----------------|--------------|-------------|
| **Regular RAG (no modifications)** | 700 | 100 | 63.7 | Decent baseline performance |
| **With LLM-generated summaries** | 700 | 100 | 67.6 | Improved retrieval, but slow index creation |
| **With Query Expansion** | 700 | 100 | 64.3 | Worse performance, longer inference time |
| **Improved summaries + larger embedding model** | 1100 | 100 | 69.5 | Higher accuracy, better summarization |
| **Best overall setup (improved summaries + large embedding model)** | 700 | 100 | 70.2 | **Best performance at this stage** |
| **With additional extensions + code snippets** | 1200 | 200 | 80.9 | Major improvement |
| **With enhanced metadata (functions & classes)** | 1200 | 200 | 84.6 | Further boost in accuracy |
| **Final optimized RAG (query fix applied)** | 1200 | 200 | 86.0 | **Best overall performance** |
| **With LLM-based re-ranker** | 1200 | 200 | 78.0 | Increased inference time and drop in accuracy |
| **With translated queries (Russian)** | 1200 | 200 | 83.3 | Handles multilingual content but slightly worse accuracy |

  

---

  


## **Final Conclusion & Next Steps**

- **Chunk size 700 with 100 overlap performed well initially, but 1200/200 provided the best final performance.**  
- **Summarizing code with an LLM improves retrieval but increases indexing time.**  
- **Switching to a larger embedding model further boosts recall.**  
- **Adding metadata and fixing query processing errors led to the highest accuracy gains.**  
- **The LLM-based re-ranker increased inference time while reducing accuracy, making it unsuitable for this task.**  
- **The system can handle multilingual content, but translating queries may slightly reduce accuracy. A hybrid approach could be explored for multilingual datasets.**  

---

**NOTE**  
The scope of this RAG system was limited to a single code base:  
[https://github.com/viarotel-org/escrcpy/blob/main/.nvmdrc](https://github.com/viarotel-org/escrcpy/blob/main/.nvmdrc).  

As a result, the tests and experiments conducted are not the best benchmarks for the techniques used. However, they still provide valuable insights.  

For future development and accuracy improvements, **a larger dataset would definitely be necessary.**