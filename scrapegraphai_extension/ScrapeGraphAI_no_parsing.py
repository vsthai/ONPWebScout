"""
ScrapeGraphAI_no_parsing.py

Contains modified GenerateAnswerNode and SmartScraperGraph classes from ScrapeGraphAI, removing the JSON format request in querying and the JSON result parsing. 

Classes:
    GenerateAnswerNodeNoParse: Modified GenerateAnswerNode to remove the request for results in json and the parsing step.
    SmartScraperGraphNoParse: Modififed SmartScraperGraph to use the GenerateAnswerNodeNoParse by default. 

All code is adapted from Scrapegraph-ai (https://github.com/ScrapeGraphAI/Scrapegraph-ai), which is licensed under the MIT license. See the LICENSE-ScrapeGraphAI file for the full text of the MIT license. 
"""

import json
from typing import List, Optional, Type
from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.nodes import GenerateAnswerNode
from scrapegraphai.utils.output_parser import get_pydantic_output_parser
from scrapegraphai.prompts import (
    TEMPLATE_CHUNKS,
    TEMPLATE_CHUNKS_MD,
    TEMPLATE_MERGE,
    TEMPLATE_MERGE_MD,
    TEMPLATE_NO_CHUNKS,
    TEMPLATE_NO_CHUNKS_MD,
)
from scrapegraphai.graphs import BaseGraph
from scrapegraphai.nodes import (
    FetchNode,
    GenerateAnswerNode,
    ParseNode,
)
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from requests.exceptions import Timeout
from tqdm import tqdm
from pydantic import BaseModel

class GenerateAnswerNodeNoParse(GenerateAnswerNode):
    """ 
    Custom GenerateAnswerNode with json response request and parsing disabled.  
    """

    def __init__(self,        
                 input: str,
                 output: List[str],
                 node_config: Optional[dict] = None,
                 node_name: str = "GenerateAnswerNoParse",
    ):
        super().__init__(input, output, node_config, node_name)

    def execute(self, state: dict) -> dict:
        """
        Executes the GenerateAnswerNode.

        Args:
            state (dict): The current state of the graph. The input keys will be used
                          to fetch the correct data from the state.

        Returns:
            dict: The updated state with the output key containing the generated answer.
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]
        user_prompt = input_data[0]
        doc = input_data[1]

        if self.node_config.get("schema", None) is not None:
            if isinstance(self.llm_model, ChatOpenAI):
                output_parser = get_pydantic_output_parser(self.node_config["schema"])
                format_instructions = output_parser.get_format_instructions()
            else:
                if not isinstance(self.llm_model, ChatBedrock):
                    output_parser = get_pydantic_output_parser(
                        self.node_config["schema"]
                    )
                    format_instructions = output_parser.get_format_instructions()
                else:
                    output_parser = None
                    format_instructions = ""
        else:
            output_parser = None
            format_instructions = ""

        if (
            not self.script_creator
            or self.force
            and not self.script_creator
            or self.is_md_scraper
        ):
            template_no_chunks_prompt = TEMPLATE_NO_CHUNKS_MD
            template_chunks_prompt = TEMPLATE_CHUNKS_MD
            template_merge_prompt = TEMPLATE_MERGE_MD
        else:
            template_no_chunks_prompt = TEMPLATE_NO_CHUNKS
            template_chunks_prompt = TEMPLATE_CHUNKS
            template_merge_prompt = TEMPLATE_MERGE

        if self.additional_info is not None:
            template_no_chunks_prompt = self.additional_info + template_no_chunks_prompt
            template_chunks_prompt = self.additional_info + template_chunks_prompt
            template_merge_prompt = self.additional_info + template_merge_prompt

        if len(doc) == 1:
            prompt = PromptTemplate(
                template=template_no_chunks_prompt,
                input_variables=["question"],
                partial_variables={
                    "context": doc,
                    "format_instructions": format_instructions,
                },
            )
            chain = prompt | self.llm_model
            if output_parser:
                chain = chain | output_parser

            try:
                answer = self.invoke_with_timeout(
                    chain, {"question": user_prompt}, self.timeout
                )
            except (Timeout, json.JSONDecodeError) as e:
                error_msg = (
                    "Response timeout exceeded"
                    if isinstance(e, Timeout)
                    else "Invalid JSON response format"
                )
                state.update(
                    {self.output[0]: {"error": error_msg, "raw_response": str(e)}}
                )
                return state

            state.update({self.output[0]: answer})
            return state

        chains_dict = {}
        for i, chunk in enumerate(
            tqdm(doc, desc="Processing chunks", disable=not self.verbose)
        ):
            prompt = PromptTemplate(
                template=template_chunks_prompt,
                input_variables=["question"],
                partial_variables={
                    "context": chunk,
                    "chunk_id": i + 1,
                    "format_instructions": format_instructions,
                },
            )
            chain_name = f"chunk{i+1}"
            chains_dict[chain_name] = prompt | self.llm_model
            if output_parser:
                chains_dict[chain_name] = chains_dict[chain_name] | output_parser

        async_runner = RunnableParallel(**chains_dict)
        try:
            batch_results = self.invoke_with_timeout(
                async_runner, {"question": user_prompt}, self.timeout
            )
        except (Timeout, json.JSONDecodeError) as e:
            error_msg = (
                "Response timeout exceeded during chunk processing"
                if isinstance(e, Timeout)
                else "Invalid JSON response format in chunk processing"
            )
            state.update({self.output[0]: {"error": error_msg, "raw_response": str(e)}})
            return state

        merge_prompt = PromptTemplate(
            template=template_merge_prompt,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": format_instructions},
        )

        merge_chain = merge_prompt | self.llm_model
        if output_parser:
            merge_chain = merge_chain | output_parser
        try:
            answer = self.invoke_with_timeout(
                merge_chain,
                {"context": batch_results, "question": user_prompt},
                self.timeout,
            )
        except (Timeout, json.JSONDecodeError) as e:
            error_msg = (
                "Response timeout exceeded during merge"
                if isinstance(e, Timeout)
                else "Invalid JSON response format during merge"
            )
            state.update({self.output[0]: {"error": error_msg, "raw_response": str(e)}})
            return state

        state.update({self.output[0]: answer})
        return state
    
class SmartScraperGraphNoParse(SmartScraperGraph):
    """
    Custom SmartScraperGraph using the GenerateAnswerNodeNoParse that removes
    json parsing. 
    """
    def __init__(
        self,
        prompt: str,
        source: str,
        config: dict,
        schema: Optional[Type[BaseModel]] = None
    ):
        super().__init__(prompt, source, config, schema)

    def _create_graph(self) -> BaseGraph:
        fetch_node = FetchNode(
            input="url | local_dir",
            output=["doc"],
            node_config={
                "llm_model": self.llm_model,
                "force": self.config.get("force", False),
                "cut": self.config.get("cut", True),
                "loader_kwargs": self.config.get("loader_kwargs", {}),
                "browser_base": self.config.get("browser_base"),
                "scrape_do": self.config.get("scrape_do"),
                "storage_state": self.config.get("storage_state"),
            },
        )
        parse_node = ParseNode(
            input="doc",
            output=["parsed_doc"],
            node_config={"llm_model": self.llm_model, "chunk_size": self.model_token},
        )

        generate_answer_node = GenerateAnswerNodeNoParse(
            input="user_prompt & (relevant_chunks | parsed_doc | doc)",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "additional_info": self.config.get("additional_info"),
                "schema": self.schema,
            },
        )
        return BaseGraph(
            nodes=[fetch_node, parse_node, generate_answer_node],
            edges=[(fetch_node, parse_node), (parse_node, generate_answer_node)],
            entry_point=fetch_node,
            graph_name=self.__class__.__name__,
        )