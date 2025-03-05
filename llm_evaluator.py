import ast
import time
import json
import re
from openai import OpenAI  #
from config import LLM_CONFIG  #

class LLMEvaluator:
    def __init__(self, model_id="gpt-4o", concurrency=50):
        """
        model_id: The name of the model used (here, OpenAI GPT-4o is used by default)
        concurrency: The number of concurrent requests (This is just a prompt. For actual concurrent control, it is necessary to combine with a thread pool or an asynchronous framework.)
        """
        self.model_id = model_id
        self.concurrency = concurrency
        # Instantiate the OpenAI client and pass in the API Key
        self.client = OpenAI(api_key=LLM_CONFIG['api_key'])

    def evaluate_path(self, path_description):
        """
        Construct the prompt according to the new evaluation criteria, and return the evaluation results after parsing the JSON data obtained by calling the interface.
        """
        prompt = f"""Please return the JSON format strictly according to the following requirements and do not include any additional instructions or code block markers:
Evaluate the path according to the Four Laws of Suspense Creation (up to 5 scenes), and please also apply the following additional evaluation criteria:

Attention Function A(PI, CI, EI)
[Dependent Factors
  - PI (Perceived Information): Considering the novelty of scene elements and the frequency of element changes.
       - Quantitative method: Count the frequency of scene switching and the number of new elements, and set a novelty score for each new element;
  - CI (cognitive information): focusing on the novelty of puzzles and the unexpectedness of clues.
       - Quantitative method: counting the puzzle solving steps and the difficulty of analyzing clues, and assigning a score to each puzzle and clue;
  - EI (Emotional Information): to examine the intensity and uniqueness of emotional elements.
       - Quantification method: Relying on the emotion dictionary or predefined emotion intensity, assigning a value to each emotion element;

[Integration method
  The above factors are weighted and summed according to certain weights to obtain the integrated score of the attention function A(PI, CI, EI).

[Emotional response function E(EI)].
  Evaluate the overall emotional response based on the intensity score of emotional words.

Please evaluate the following path descriptions in strict accordance with the above requirements and the Four Laws of Creation.
And return the data in JSON format containing three parts: scene score, violation check, and improvement suggestion.

path description: {path_description}
"""
        print("DEBUG: The ID of the model used:", self.model_id)
        response = self._safe_create_completion(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional assistant focused on the assessment of Suspense Creation and need to provide scoring and suggestions for improvement for each scene in conjunction with the Four Laws of Suspense Creation and the added Attention and Emotion functions."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        # If other texts are mixed in the returned data, extract the JSON part through _parse_response.
        return self._parse_response(response.choices[0].message)

    def direct_generate(self, prompt: str) -> str:
        """Bypass the evaluation system and directly generate the original text."""
        response = self._safe_create_completion(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "You are an expert in suspense novel creation.
"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # Higher creativity
        )
        return response.choices[0].message.content.strip()   # Directly return plain text.

    def _safe_create_completion(self, **kwargs):
        max_retries = LLM_CONFIG.get("max_retries", 3)
        for i in range(max_retries):
            try:
                # Call the new version of the interface: Make the call through the client in the instance.
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                wait_time = 1 * (2 ** i)
                print(f"Request exception: {e}. Retrying after waiting for {wait_time} seconds...")
                time.sleep(wait_time)
        raise Exception("重试次数超过限制，请稍后再试。")

    def _parse_response(self, content):
        # If the content is not a string, try to obtain its content attribute.
        if not isinstance(content, (str, bytes, bytearray)):
            try:
                content_str = content.content
            except AttributeError:
                content_str = str(content)
        else:
            content_str = content
        print("原始返回内容:", content_str)

        # Try to find the code block in JSON format. First, look for json ..., and then look for python ....
        match = re.search(r"```json\s*(\{.*\})\s*```", content_str, re.DOTALL)
        if not match:
            match = re.search(r"```python\s*(\{.*\})\s*```", content_str, re.DOTALL)
        if match:
            data_str = match.group(1)
            try:
                # Give priority to using json.loads for parsing.
                result = json.loads(data_str)
                return result
            except json.JSONDecodeError:
                print("Error: The extracted content is not valid JSON. Try to parse it using ast.literal_eval().")
                try:
                    result = ast.literal_eval(data_str)
                    return result
                except Exception as e:
                    print(f"Error: The parsing by ast.literal_eval() has failed: {e}")
                    return data_str

        # If all the previous matches fail, then perform a downgraded processing: Extract all substrings that seem to be dictionaries from the string.
        def extract_dict_candidates(text):
            candidates = []
            start = text.find('{')
            while start != -1:
                balance = 0
                found = False
                for i in range(start, len(text)):
                    if text[i] == '{':
                        balance += 1
                    elif text[i] == '}':
                        balance -= 1
                        if balance == 0:
                            candidates.append(text[start:i + 1])
                            start = text.find('{', i + 1)
                            found = True
                            break
                if not found:
                    break
            return candidates

        candidates = extract_dict_candidates(content_str)
        if not candidates:
            print("错误: 没有从返回内容中提取到有效的字典对象。")
            return content_str

        # Error: No valid dictionary object was extracted from the returned content.
        for candidate in candidates:
            try:
                result = ast.literal_eval(candidate)
                return result
            except Exception as e:
                print(f"Error: Unable to parse the candidate string: {candidate}, Exception: {e}")
        return content_str

if __name__ == "__main__":
    evaluator = LLMEvaluator(model_id="gpt-4o", concurrency=50)
    test_description = "This is a test plot."