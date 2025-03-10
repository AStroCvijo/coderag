
# **Evaluation of Retrieval-Augmented Generation (RAG) for Code Search**

## **1. Baseline Performance (Regular RAG with No Modifications)**
The initial setup involved splitting the code into chunks and embedding them using **text-embedding-small**. Using similarity retrieval, this approach achieved the following:  
- **Best-performing setup:** **Chunk size 700, chunk overlap 100**  
- **Recall@10:** **63.7%**  

✅ **Decent performance, but significant room for improvement.**  

---

## **2. Using LLM Summaries for Embeddings**
Since queries are in **natural language**, I experimented with **summarizing the code using an LLM** before embedding. While this made **vector store creation much slower**, it improved retrieval accuracy:  
- **Chunk size 700, overlap 100 → Recall@10: 67.6%**  

✅ **Better than the baseline, but still far from optimal.**  

---

## **3. Experimenting with Query Expansion (Did Not Work Well)**
Next, I attempted to **expand the queries** using different prompt templates. However, this approach had **several drawbacks**:  
- **Inference time increased significantly.**  
- **Accuracy dropped across all chunk sizes and overlaps.**  
- **Best attempt (shortened expanded queries) reached only Recall@10: 64.3%, still worse than the original queries.**  

❌ **Conclusion: Query expansion is not worth it for this task.**  

---

## **4. Improving Summarization & Using a Bigger Embedding Model**
To further enhance retrieval, I:  
- Used **a better summarization prompt with more metadata**.  
- Switched from **text-embedding-small** to **text-embedding-large** (slightly more expensive but potentially more effective).  

Results:  
- **Chunk size 1100, overlap 100 → Recall@10: 69.5%** (Best for this setting)  
- **Chunk size 700, overlap 100 → Recall@10: 70.2%** (Best overall)  

✅ **This was the most effective setup, achieving the highest retrieval accuracy.**  

---

## **5. Further Enhancements & Best Achieved Performance**
After noticing missing file extensions in the dataset, I included them, which improved accuracy. I then experimented with **adding code snippets alongside the summaries**, leading to a major improvement:  
- **Chunk size 1200, overlap 200 → Recall@10: 80.9%**  

To push accuracy even further, I enhanced metadata by including **a list of functions and classes present in each file**, which resulted in:  
- **Recall@10: 84.6%**  

Finally, I found an error in the querying process where files were first sorted by relevance before ensuring uniqueness in the top 10, potentially causing crucial losses. Fixing this boosted accuracy to:  
- **Recall@10: 86.0%** (Best achieved performance)  
- **Recall@40: 90.3%** (Suggesting a re-ranking system could further improve results)  

✅ **Best RAG system so far, with a significant improvement over the baseline.**  

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

---

## **Final Conclusion & Next Steps**
- **Chunk size 700 with 100 overlap performed well initially, but 1200/200 provided the best final performance.**  
- **Summarizing code with an LLM improves retrieval but increases indexing time.**  
- **Switching to a larger embedding model further boosts recall.**  
- **Adding metadata and fixing query processing errors led to the highest accuracy gains.**  
- **With Recall@40 of 90.3%, a re-ranking system could be a promising next step.**  

🚀 **This optimized RAG pipeline significantly outperforms the baseline, with more room for future enhancements.**

