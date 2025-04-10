import json
from typing import TypedDict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph
from langchain.output_parsers import PydanticOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function


class ChunkState(TypedDict):
    """
    TypedDict for the state of the chunk in the LangGraph workflow.
    """
    chunk: str


class PairListState(ChunkState):
    """
    TypedDict for the state of the instruction-response pairs in the LangGraph workflow.
    """
    pairs: list[dict]


class InstructionResponse(BaseModel):
    """
    Pydantic model for instruction-response pairs.
    """
    instruction: str
    response: str


class MedicalSFTDatasetBuilder:
    def __init__(
            self,
            pdf_path: str,
            model_name: str = "gpt-4o",
            temperature: float = 0.0,
            chunk_size: int = 500,
            chunk_overlap: int = 150,
            debug: bool = False
    ):
        """
        :param pdf_path: Path to the medical textbook PDF
        :param model_name: LLM model name (e.g., gpt-4)
        :param temperature: Sampling temperature for the LLM
        :param chunk_size: Size of text chunks to split
        :param chunk_overlap: Overlap between chunks
        """
        self.pdf_path = Path(pdf_path).expanduser().resolve()
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = PromptTemplate.from_template("""
        You are a helpful assistant that extracts useful training data for a language model.

        From the following medical text, generate a list of instruction-response training examples (max 5).

        Return ONLY a JSON array of objects, where each object has:
        - instruction: a prompt to teach the model
        - response: a useful answer from the passage

        Text:
        {chunk}

        Return only the JSON list.
        """)
        self.parser = PydanticOutputParser(pydantic_object=InstructionResponse)
        self.function_schema = [convert_to_openai_function(InstructionResponse)]
        self.graph = self._build_langgraph()
        self.debug = debug

    def _build_langgraph(self) -> CompiledGraph:
        """
        Build the LangGraph workflow for generating instruction-response pairs.

        :return: The compiled LangGraph workflow
        """
        workflow = StateGraph(state_schema=PairListState)
        workflow.add_node("generate_pair", self._generate_instruction_response)
        workflow.set_entry_point("generate_pair")
        workflow.set_finish_point("generate_pair")
        return workflow.compile()

    def _load_and_chunk_pdf(self) -> Document:
        """
        Load the PDF and split it into chunks for processing.

        :return: The list of text chunks
        """
        loader = PyMuPDFLoader(str(self.pdf_path))
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return splitter.split_documents(documents)

    @retry(
        wait=wait_exponential(min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(ValueError)
    )
    def _safe_invoke(self, chunk_text):
        prompt_text = self.prompt.format(chunk=chunk_text)
        if self.debug:
            print("\n--- Prompting LLM ---")
            print(prompt_text)
        result = self.llm.invoke([HumanMessage(content=prompt_text)])
        if self.debug:
            print("\n--- Inovking LLM ---")
            print(result)
        try:
            json_parse = json.loads(result.content)
            return json_parse
        except Exception as e:
            print("JSON parsing failed:", str(e))
            raise ValueError("Failed to parse LLM output")

    def _generate_instruction_response(self, state: ChunkState) -> PairListState:
        """
        Generate instruction-response pairs from the given chunk of text.

        :param state: The state containing the chunk of text
        :return: The state with generated instruction-response pairs
        """
        if self.debug:
            print("LangGraph node triggered")
        try:
            parsed = self._safe_invoke(state["chunk"])
            if self.debug:
                print("Parsed pairs:", parsed)
            return {"pairs": parsed}
        except Exception as e:
            print("Exception in node:", e)
            return {"pairs": [{
                "instruction": "ERROR",
                "response": f"Failed after retries: {str(e)}"
            }]}

    def build_dataset(
            self,
            output_parquet: str = None,
            output_jsonl: str = None,
            limit=None
    ) -> None:
        chunks = self._load_and_chunk_pdf()
        results = []

        for i, chunk in enumerate(tqdm(chunks[:limit] if limit else chunks, desc="Generating pairs")):
            chunk: Document
            state = {"chunk": chunk.page_content}
            result = self.graph.invoke(state)
            response_list = result.get("pairs", [])
            for j, response in enumerate(response_list):
                results.append({
                    "id": f"sft-{i}-{j}",
                    "instruction": response.get("instruction", ""),
                    "response": response.get("response", ""),
                    "source_page": chunk.metadata.get("page", None)
                })

        df = pd.DataFrame(results)

        if output_parquet:
            df.to_parquet(output_parquet, index=False)

        if output_jsonl:
            if "instruction" in df.columns and "response" in df.columns:
                df[["instruction", "response"]].to_json(output_jsonl, orient="records", lines=True)
            else:
                print("Skipped JSONL export: Missing required columns.")


if __name__ == "__main__":
    builder = MedicalSFTDatasetBuilder("./Harrisons_Manual_of_Medicine_18th_Edition_2.pdf")
    builder.build_dataset(
        output_parquet="sft_medical_dataset.parquet",
        output_jsonl="sft_medical_dataset.jsonl",
        limit=100
    )