# AskAPhilosopher-
A RAG-LLM based chat-bot-y thingy to converse with ideas of philosophers.

Using streamlit, LLM and Faiss vector database to inquire about ideas of different philosophers.

Information about various philosophers was gathered and then summarised with GPT. Summarised Information is used to create a vector database for individual philosophers. LLM (PaLM or local-Llama) are used to retrieve and generate answers, from the database, to the query submitted by user.

More philosophers can be added, by providing a relevant summary file.

(P.S. More than chatting to ideas-of-philosophers, it is closer to chatting with the interpretations of the said philosphers. However, because of semantic learning in Transformer architecture it does capture, to some extent, the sense which the ideas point towards.)
