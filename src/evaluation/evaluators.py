"""LangSmith evaluation utilities."""
from typing import Optional, Callable
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_settings


class PipelineEvaluator:
    """Evaluator for pipeline responses using LangSmith."""
    
    def __init__(self):
        """Initialize the evaluator."""
        settings = get_settings()
        
        self._client = Client()
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0
        )
    
    def create_relevance_evaluator(self) -> LangChainStringEvaluator:
        """Create an evaluator for response relevance.
        
        Returns:
            LangChainStringEvaluator for relevance
        """
        return LangChainStringEvaluator(
            "criteria",
            config={
                "criteria": {
                    "relevance": "Is the response relevant to the user's question?"
                },
                "llm": self._llm
            }
        )
    
    def create_helpfulness_evaluator(self) -> LangChainStringEvaluator:
        """Create an evaluator for response helpfulness.
        
        Returns:
            LangChainStringEvaluator for helpfulness
        """
        return LangChainStringEvaluator(
            "criteria",
            config={
                "criteria": {
                    "helpfulness": "Does the response provide helpful and actionable information?"
                },
                "llm": self._llm
            }
        )
    
    def create_coherence_evaluator(self) -> LangChainStringEvaluator:
        """Create an evaluator for response coherence.
        
        Returns:
            LangChainStringEvaluator for coherence
        """
        return LangChainStringEvaluator(
            "criteria",
            config={
                "criteria": {
                    "coherence": "Is the response well-structured and easy to understand?"
                },
                "llm": self._llm
            }
        )
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        reference: Optional[str] = None
    ) -> dict:
        """Evaluate a single response.
        
        Args:
            query: User's input query
            response: Model's response
            reference: Optional reference answer
            
        Returns:
            Dictionary with evaluation scores
        """
        relevance_eval = self.create_relevance_evaluator()
        helpfulness_eval = self.create_helpfulness_evaluator()
        coherence_eval = self.create_coherence_evaluator()
        
        input_dict = {"input": query}
        
        relevance_result = relevance_eval.evaluate_strings(
            prediction=response,
            input=query,
            reference=reference
        )
        
        helpfulness_result = helpfulness_eval.evaluate_strings(
            prediction=response,
            input=query,
            reference=reference
        )
        
        coherence_result = coherence_eval.evaluate_strings(
            prediction=response,
            input=query,
            reference=reference
        )
        
        return {
            "relevance": {
                "score": relevance_result.get("score", 0),
                "reasoning": relevance_result.get("reasoning", "")
            },
            "helpfulness": {
                "score": helpfulness_result.get("score", 0),
                "reasoning": helpfulness_result.get("reasoning", "")
            },
            "coherence": {
                "score": coherence_result.get("score", 0),
                "reasoning": coherence_result.get("reasoning", "")
            }
        }
    
    def create_rag_faithfulness_evaluator(self):
        """Create evaluator for RAG faithfulness.
        
        Checks if the response is grounded in the retrieved context.
        
        Returns:
            Custom evaluator function
        """
        prompt = ChatPromptTemplate.from_template("""
You are evaluating whether an AI response is faithful to the provided context.

Context: {context}
Question: {question}
Response: {response}

Evaluate on a scale of 1-5:
1 = Completely unfaithful, contains hallucinations
2 = Mostly unfaithful, some accurate parts
3 = Partially faithful, mix of grounded and ungrounded claims
4 = Mostly faithful, minor issues
5 = Completely faithful, all claims grounded in context

Provide:
- Score (1-5)
- Brief reasoning

Format: Score: X | Reasoning: ...
""")
        
        chain = prompt | self._llm
        
        def evaluate_faithfulness(context: str, question: str, response: str) -> dict:
            result = chain.invoke({
                "context": context,
                "question": question,
                "response": response
            })
            
            result_text = result.content
            try:
                score_part = result_text.split("|")[0]
                score = int(score_part.split(":")[1].strip())
                reasoning = result_text.split("|")[1].split(":")[1].strip()
            except (IndexError, ValueError):
                score = 0
                reasoning = result_text
            
            return {
                "score": score / 5,  # Normalize to 0-1
                "reasoning": reasoning
            }
        
        return evaluate_faithfulness


def get_langsmith_trace_url(run_id: str) -> str:
    """Get the LangSmith trace URL for a run.
    
    Args:
        run_id: The run ID from LangSmith
        
    Returns:
        URL to the LangSmith trace
    """
    settings = get_settings()
    project = settings.langchain_project
    return f"https://smith.langchain.com/o/default/projects/p/{project}/r/{run_id}"
