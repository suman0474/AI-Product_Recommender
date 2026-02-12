import logging
import os
import re
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts_library import load_prompt_sections
# Import directly from the module
from advanced_parameters import discover_advanced_parameters

logger = logging.getLogger(__name__)

class SalesWorkflowService:
    """
    Central service for handling Sales Agent workflow logic.
    Used by both the API endpoint and LangChain tools to ensure consistent behavior.
    """

    def __init__(self):
        self._load_prompts()
    
    def _load_prompts(self):
        """Load specific prompt sections for the sales workflow"""
        self.prompts = load_prompt_sections("sales_agent_prompts", default_section="SALES_CONSULTANT")
        
        # Map prompts for easy access
        self.PROMPTS = {
            "CONSULTANT": self.prompts["SALES_CONSULTANT"],
            "FRESH_CONVERSATION": self.prompts["FRESH_CONVERSATION"],
            "REQUIREMENTS_PROVIDED": self.prompts["REQUIREMENTS_PROVIDED"],
            "ASK_MISSING_FIELDS": self.prompts["ASK_MISSING_FIELDS"],
            "CONFIRM_AFTER_MISSING_INFO": self.prompts["CONFIRM_AFTER_MISSING_INFO"],
            "SHOW_ADDITIONAL_SPECS": self.prompts["SHOW_ADDITIONAL_SPECS"],
            "DECLINE_ADDITIONAL_SPECS": self.prompts["DECLINE_ADDITIONAL_SPECS"],
            "NO_ADDITIONAL_SPECS": self.prompts["NO_ADDITIONAL_SPECS"],
            "ADVANCED_PARAMS_YES": self.prompts["ADVANCED_PARAMS_YES"],
            "ADVANCED_PARAMS_NO": self.prompts["ADVANCED_PARAMS_NO"],
            "SHOW_SUMMARY": self.prompts["SHOW_SUMMARY"],
            "FINAL_ANALYSIS": self.prompts["FINAL_ANALYSIS"],
        }

    def _get_llm(self):
        """Initialize LLM with standard configuration"""
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError("LLM component not ready")

    def _format_available_parameters(self, params) -> str:
        """Format parameters for display in prompts"""
        formatted = []
        for param in params:
            if isinstance(param, dict):
                name = param.get('name') or param.get('key') or (list(param.keys())[0] if param else '')
            else:
                name = str(param).strip()
            name = name.replace('_', ' ')
            name = re.split(r'[\(\[\{]', name, 1)[0].strip()
            name = " ".join(name.split())
            name = name.title()
            formatted.append(f"- {name}")
        return "\n".join(formatted)

    def process_step(self, step: str, user_message: str, data_context: Dict[str, Any], 
                     intent: str, search_session_id: str) -> Dict[str, Any]:
        """
        Process a single step of the sales workflow.
        Returns dictionary with 'content' (response text) and 'nextStep'.
        """
        llm = self._get_llm()
        
        # Handle knowledge questions
        if intent == "knowledgeQuestion":
            context_hint = "How can I help you with your product selection?"
            prompt_template = self.PROMPTS["CONSULTANT"].format(
                user_message=user_message,
                context_hint=context_hint
            )
            full_prompt = ChatPromptTemplate.from_template(prompt_template)
            response_chain = full_prompt | llm | StrOutputParser()
            llm_response = response_chain.invoke({"user_input": user_message})
            
            return {
                "content": llm_response,
                "nextStep": step,
                "maintainWorkflow": True
            }

        prompt_template = ""
        next_step = step
        
        # --- Step Logic ---
        
        if step == 'initialInput':
            product_type = data_context.get('productType', 'a product')
            prompt_template = self.PROMPTS["FRESH_CONVERSATION"].format(
                search_session_id=search_session_id,
                product_type=product_type
            )
            next_step = "awaitAdditionalAndLatestSpecs"
        
        elif step == 'initialInputWithSpecs':
            product_type = data_context.get('productType', 'a product')
            prompt_template = self.PROMPTS["REQUIREMENTS_PROVIDED"].format(
                product_type=product_type
            )
            next_step = "awaitAdditionalAndLatestSpecs"
        
        elif step == 'askForMissingFields':
            product_type = data_context.get('productType', 'your product')
            missing_fields = data_context.get('missingFields', '')
            prompt_template = self.PROMPTS["ASK_MISSING_FIELDS"].format(
                product_type=product_type,
                missing_fields=missing_fields
            )
            next_step = "awaitMissingInfo"
        
        elif step == 'confirmAfterMissingInfo':
            product_type = data_context.get('productType', 'a product')
            prompt_template = self.PROMPTS["CONFIRM_AFTER_MISSING_INFO"].format(
                product_type=product_type
            )
            next_step = "awaitAdditionalAndLatestSpecs"
        
        elif step == 'awaitAdditionalAndLatestSpecs':
            user_lower = user_message.lower().strip()
            product_type = data_context.get('productType')
            
            affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']
            is_yes = any(k in user_lower for k in affirmative_keywords)
            
            if is_yes:
                try:
                    parameters_result = discover_advanced_parameters(product_type)
                    remaining_parameters = parameters_result.get('unique_specifications', [])[:15]
                except Exception as e:
                    logger.warning(f"Could not discover parameters: {e}")
                    remaining_parameters = []
                
                if remaining_parameters:
                    params_display = self._format_available_parameters(remaining_parameters)
                    prompt_template = self.PROMPTS["SHOW_ADDITIONAL_SPECS"].format(
                        product_type=product_type,
                        params_display=params_display
                    )
                else:
                    prompt_template = self.PROMPTS["NO_ADDITIONAL_SPECS"].format(
                        product_type=product_type
                    )
                    next_step = "awaitAdvancedSpecs"
            else:
                prompt_template = self.PROMPTS["DECLINE_ADDITIONAL_SPECS"].format(
                     product_type=product_type
                )
                next_step = "awaitAdvancedSpecs"
                
        elif step == 'awaitAdvancedSpecs':
            user_lower = user_message.lower().strip()
            affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']
            is_yes = any(k in user_lower for k in affirmative_keywords)
            
            if is_yes:
                 prompt_template = self.PROMPTS["ADVANCED_PARAMS_YES"]
            else:
                 prompt_template = self.PROMPTS["ADVANCED_PARAMS_NO"]
                 next_step = "showSummary"

        elif step == 'showSummary':
             prompt_template = self.PROMPTS["SHOW_SUMMARY"]

        elif step == 'finalAnalysis':
             prompt_template = self.PROMPTS["FINAL_ANALYSIS"]
        
        # Fallback
        if not prompt_template:
            prompt_template = self.PROMPTS["CONSULTANT"].format(
                user_message=user_message,
                context_hint=f"Current step: {step}"
            )

        # Generate Response
        full_prompt = ChatPromptTemplate.from_template(prompt_template)
        response_chain = full_prompt | llm | StrOutputParser()
        llm_response = response_chain.invoke({})

        return {
            "content": llm_response,
            "nextStep": next_step
        }
