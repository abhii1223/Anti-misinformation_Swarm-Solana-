import requests
import json
from typing import Dict, Any, Tuple

class FactDetector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def check_fact(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Check if a statement is factual and return detailed analysis.
        
        Args:
            text (str): The statement to check
            
        Returns:
            Tuple containing:
            - str: "TRUTH" or "FAKE"
            - float: Confidence score (0.0 to 1.0)
            - Dict: Detailed analysis
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        prompt = f"""Analyze the following statement for factual accuracy and provide a detailed assessment:
        
        {text}
        
        Please provide your response in the following JSON format:
        {{
            "verdict": "TRUTH" or "FAKE",
            "confidence": number between 0.0 and 1.0,
            "explanation": "brief explanation of why",
            "evidence": ["list of supporting evidence or sources"],
            "red_flags": ["list of any suspicious elements or inconsistencies"],
            "fact_score": 1 if completely factual, 0 if completely false, or decimal for partial truth
        }}
        
        Important: The confidence and fact_score must be numbers between 0.0 and 1.0.
        """
        
        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "You are a fact-checking assistant specializing in detecting manipulated content and deepfakes. Provide detailed analysis in JSON format. Always include numerical confidence and fact scores."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            if response.status_code != 200:
                return "ERROR", 0.0, {"error": f"API request failed - {response.status_code}"}

            result = response.json()
            if 'choices' not in result or len(result['choices']) == 0:
                return "ERROR", 0.0, {"error": "Invalid API response"}

            content = result['choices'][0]['message']['content'].strip()
            try:
                # Try to find JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    analysis = json.loads(json_str)
                else:
                    analysis = {"verdict": "UNKNOWN", "confidence": 0.0, "fact_score": 0.0}
                
                # Ensure we have valid scores
                verdict = analysis.get('verdict', 'UNKNOWN')
                confidence = float(analysis.get('confidence', 0.0))
                fact_score = float(analysis.get('fact_score', 0.0))
                
                # Validate scores
                confidence = max(0.0, min(1.0, confidence))
                fact_score = max(0.0, min(1.0, fact_score))
                
                # Calculate final score
                final_score = (confidence + fact_score) / 2
                
                # Ensure we have a valid verdict
                if verdict not in ['TRUTH', 'FAKE']:
                    verdict = 'TRUTH' if fact_score > 0.5 else 'FAKE'
                
                return verdict, final_score, analysis
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing response: {str(e)}")
                print(f"Raw content: {content}")
                return "ERROR", 0.0, {"error": "Could not parse analysis", "raw_response": content}

        except Exception as e:
            return "ERROR", 0.0, {"error": str(e)}

def main():
    api_key = "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"
    detector = FactDetector(api_key)
    
    print("\n=== Deepfake and Fact-Checking Analysis Tool ===")
    print("Enter a statement to analyze its authenticity and factual accuracy.")
    print("Type 'quit' to exit.\n")
    
    while True:
        text = input("\nEnter the statement to analyze: ")
        if text.lower() == 'quit':
            break
            
        verdict, confidence, analysis = detector.check_fact(text)
        
        print("\n=== Analysis Results ===")
        print(f"Statement: {text}")
        print(f"Verdict: {verdict}")
        print(f"Confidence Score: {confidence:.2f}")
        
        if 'explanation' in analysis:
            print(f"\nExplanation: {analysis['explanation']}")
        
        if 'evidence' in analysis and analysis['evidence']:
            print("\nSupporting Evidence:")
            for evidence in analysis['evidence']:
                print(f"- {evidence}")
        
        if 'red_flags' in analysis and analysis['red_flags']:
            print("\nPotential Red Flags:")
            for flag in analysis['red_flags']:
                print(f"- {flag}")
        
        print(f"\nFact Score: {analysis.get('fact_score', 0.0):.2f}")
        
        # Additional analysis for potential deepfake detection
        if confidence < 0.5 or analysis.get('fact_score', 0.0) < 0.5:
            print("\n⚠️ Warning: This content shows characteristics of potential manipulation or deepfake content.")
            print("Consider verifying through additional sources.")

if __name__ == "__main__":
    main()

# import streamlit as st
# import requests
# import json

# class FactDetector:
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#         self.base_url = "https://api.perplexity.ai"
#         self.headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }

#     def check_fact(self, text: str) -> str:
#         endpoint = f"{self.base_url}/chat/completions"
        
#         prompt = f"""Is the following statement true or false? Answer only with "Truth" or "Fake":
        
#         {text}
#         """
        
#         payload = {
#             "model": "sonar",
#             "messages": [
#                 {"role": "system", "content": "You are a fact-checking assistant. Answer only with 'Truth' or 'Fake'."},
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": 0.1,
#             "max_tokens": 10
#         }
        
#         try:
#             response = requests.post(endpoint, headers=self.headers, json=payload)
#             if response.status_code != 200:
#                 return f"❌ Error: API request failed - {response.status_code}"

#             result = response.json()
#             if 'choices' not in result or len(result['choices']) == 0:
#                 return "❌ Error: Invalid API response"

#             answer = result['choices'][0]['message']['content'].strip().lower()
#             if 'truth' in answer or 'true' in answer:
#                 return "✅ TRUTH"
#             elif 'fake' in answer or 'false' in answer:
#                 return "❌ FAKE"
#             else:
#                 return f"⚠️ Could not determine: {answer}"

#         except Exception as e:
#             return f"❌ Error: {str(e)}"

# def main():
#     st.set_page_config(page_title="Truth or Fake", page_icon="🔍")
#     st.title("🔍 Truth or Fake Checker")
#     st.markdown("Enter any statement to check if it's true or fake.")
    
#     # Hardcoded API key
#     api_key = "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"
    
#     detector = FactDetector(api_key)
    
#     # Text input box
#     statement = st.text_area("Enter your statement here:", height=150)
    
#     if st.button("Check Statement"):
#         if not statement:
#             st.warning("Please enter a statement to check.")
#         else:
#             with st.spinner("Analyzing..."):
#                 result = detector.check_fact(statement)
#                 if "TRUTH" in result:
#                     st.success(result)
#                 elif "FAKE" in result:
#                     st.error(result)
#                 else:
#                     st.warning(result)

# if __name__ == "__main__":
#     main()
