from abc import ABC, abstractmethod

# 1. Abstract Base Class
class Runnable(ABC):
    """
    Abstract base class for all runnables.
    This defines the valid interface (invoke) that all components must implement.
    This mimics the core 'Runnable' protocol in LangChain.
    """
    @abstractmethod
    def invoke(self, input_data):
        pass

# 2. Prompt Template Component
class NakliPromptTemplate(Runnable):
    """
    A simple prompt template that formats a string with input variables.
    Equivalent to LangChain's PromptTemplate.
    """
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict: dict) -> str:
        # Formats the template string using the provided dictionary
        # This implementation simply replaces placeholders in the string
        print(f"DEBUG: Formatting prompt with arguments: {input_dict}")
        return self.template.format(**input_dict)

# 3. LLM Component
class NakliLLM(Runnable):
    """
    A mocked LLM that returns a static response based on the input prompt.
    Equivalent to a ChatOpenAI or similar LLM wrapper.
    """
    def invoke(self, prompt: str) -> dict:
        # Simulates a model response based on the input prompt
        print(f"DEBUG: Sending prompt to LLM: '{prompt}'")
        # Simple logic to simulate a relevant response
        if "poem" in prompt.lower():
            return {"response": f"Here is a poem about the topic mention in prompt: {prompt}"}
        elif "fact" in prompt.lower():
             return {"response": f"Here is a fact based on: {prompt}"}
        else:
            return {"response": f"I received your prompt: {prompt}"} 

# 4. Connector (Chain) Component
class RunnableConnector(Runnable):
    """
    A connector that chains multiple Runnables together sequentially.
    Equivalent to the '|' pipe operator chain or specific chain classes in LangChain (RunnableSequence).
    """
    def __init__(self, runnable_list: list[Runnable]):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        # Passes the output of one runnable as the input to the next
        current_input = input_data
        for runnable in self.runnable_list:
            current_input = runnable.invoke(current_input)
        return current_input

# Demonstration of the concept
if __name__ == "__main__":
    # Create the components
    # 1. Start with a prompt template
    template = NakliPromptTemplate(
        template="Write a {length} poem about {topic}",
        input_variables=['length', 'topic']
    )
    
    # 2. Add an LLM
    llm = NakliLLM()
    
    # 3. Connect them into a Chain: Template -> LLM
    chain = RunnableConnector([template, llm])
    
    # Invoke the chain
    input_data = {'length': 'short', 'topic': 'india'}
    print(f"\nInvoking chain with input: {input_data}")
    
    response = chain.invoke(input_data)
    
    print("\nFinal Response:")
    print(response)
