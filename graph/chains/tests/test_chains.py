from dotenv import load_dotenv
from pprint import pprint

load_dotenv()


from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader


from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.router import RouteQuery, question_router




def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    # Pass the retrieved document for grading
    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_txt})

    # Check if the grading was correct
    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = ""
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    # Pass the retrieved document for grading
    res: GradeDocuments = retrieval_grader.invoke({"question": "How to make pizza" , "document": doc_txt})

    # Check if the grading was correct
    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)

# def test_hallucination_grader_answer_yes() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)
#
#     generation = generation_chain.invoke({"context": docs, "question": question})
#     res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": generation})
#
#     assert res.binary_score
#
# def test_hallucination_grader_answer_no() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)
#     res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": "In order to make pizza we need to first start with the dough"})
#
#     assert not res.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "How to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"




