from services.internal_classes import Request
from services.llm_service.service import LlmService


LLM_SERVICE = LlmService()


class Service:
    def __init__(self) -> None:
        pass

    def apply(self, request: Request) -> None:
        """
        - Parses the request
        - Passes the prompt to the LLM Model
        - Get output and then pass it upstream
        """
        llm_service_response = LLM_SERVICE.apply(request=request)
        return llm_service_response
