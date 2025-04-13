import os
import openai
import requests, aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict
from uagents import Agent, Context, Model

# Agent Addresses
OPEN_AI_KEY = 'OPEN_AI_KEY'
API_KEY = "FACTCHECK_API_KEY"
TRUSTED_SOURCES = [
    "Snopes",
    "PolitiFact",
    "FactCheck.org",
    "Washington Post",
    "AP Fact Check",
    "Reuters Fact Check",
    "BBC Reality Check",
    "Full Fact",
    "The Guardian Fact Check",
    "Center for Responsive Politics",
    "Lead Stories",
    "AFP Fact Check",
    "Poynter",
    "The New York Times Fact Check",
    "USA Today Fact Check"
]


# Request model
class requestDeepTruthAgent(Model):
    query: str
    sender_address: str

# Response model (contains misinformation or information)
class responseDeepTruthAgent(Model):
    result: str  # Contains either 'misinformation' or 'information'
    source: str  # Source of the classification

# Define the main agent
mainAgent = Agent(
    name='misinformation_classification_agent',
    port=5068,
    endpoint='http://localhost:5068/submit',
    seed='misinformation_classification_seed'
)

# Track received responses
received_responses = []

async def check_information(query: str) -> str:
    """
    This function checks if the given statement is misinformation or information
    by querying the Google Fact Check API.
    If no recent results (<7 days old) are found, it queries OpenAI.
    """
    print(f'In check_information')

    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={API_KEY}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f'Google Response Status Code: {response.status}')
            
            if response.status == 200:
                data = await response.json()
                claims = data.get("claims", [])
                recent_claims = []
                one_week_ago = datetime.utcnow() - timedelta(days=30)

                # for claim in claims:
                #     claim_date_str = claim.get("claimDate", "")
                    
                #     if claim_date_str:
                #         claim_date = datetime.strptime(claim_date_str, "%Y-%m-%dT%H:%M:%SZ")
                        
                #         if claim_date >= one_week_ago:
                #             recent_claims.append(claim)

                if not claims:
                    return check_with_openai(query)

                for claim in claims[:5]:  # Get top 5 recent results
                    claimant = claim.get("claimant", "Unknown source")
                    review = claim.get("claimReview", [{}])[0]
                    rating = review.get("textualRating", "No rating available")
                    review_url = review.get("url", "No URL available")
                    
                    if "false" in rating.lower():
                        return f"misinformation from {claimant} | Source: {review_url}"
                    elif "true" in rating.lower():
                        return f"information from {claimant} | Source: {review_url}"
                
                return "No definitive rating found from recent claims."
            else:
                return check_with_openai(query) 

# async def check_information(query: str, check_openai: bool = False) -> str:
#     """
#     This function checks if the given statement is misinformation or information
#     by querying the Google Fact Check API and filtering results by trusted sources.
#     """
#     print(f'Checking information for: {query}')

#     url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={API_KEY}"
#     if not check_openai:
#         async with aiohttp.ClientSession() as session:
#             async with session.get(url) as response:
#                 print(f'Google Response Status Code: {response.status}')
                
#                 if response.status == 200:
#                     data = await response.json()
#                     claims = data.get("claims", [])
                    
#                     if not claims:
#                         return check_with_openai(query)

#                     for claim in claims:
#                         review = claim.get("claimReview", [{}])[0]
#                         publisher_name = review.get("publisher", {}).get("name", "")

#                         # Filter results by trusted sources
#                         if publisher_name in TRUSTED_SOURCES:
#                             rating = review.get("textualRating", "").lower()
#                             review_url = review.get("url", "No URL available")

#                             if "false" in rating:
#                                 return f"misinformation from {publisher_name} | Source: {review_url}"
#                             elif "true" in rating:
#                                 return f"information from {publisher_name} | Source: {review_url}"

#                     return "Not Found: No definitive rating found from trusted sources."
#     if check_openai:        
#         return check_with_openai(query)

def check_with_openai(query: str) -> str:
    """
    This function queries OpenAI to check if the given statement is misinformation or information.
    """
    openai.api_key = OPEN_AI_KEY

    # Crafting the message for OpenAI Chat Completion API
    messages = [
        {"role": "system", "content": "You are an assistant that classifies statements as either 'misinformation' or 'information' based on the content."},
        {"role": "user", "content": f"Is the following statement true or false? Please reply with 'misinformation' or 'information': {query}"}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # You can use other models such as "gpt-3.5-turbo"
            messages=messages,
            max_tokens=60,
            temperature=0.0,
        )
        print('-' * 100)
        print(response)
        print('-' * 100)
        answer = response['choices'][0]['message']['content'].strip().lower()
        print('-' * 100)
        print(answer)
        print('-' * 100)
        
        if "misinformation" in answer:
            return f"misinformation"
        elif "information" in answer:
            return f"information"
        else:
            return "OpenAI could not classify this statement."
    except Exception as e:
        return f"Error querying OpenAI: {str(e)}"

@mainAgent.on_event('startup')
async def startup_handler(ctx: Context):
    ctx.logger.info(f'Agent {ctx.agent.name} started at {ctx.agent.address}')

# Handler for receiving query
@mainAgent.on_message(model=requestDeepTruthAgent)
async def handle_query(ctx: Context, sender: str, msg: requestDeepTruthAgent):
    global received_responses
    received_responses = []  # Reset for each new request
    ctx.logger.info(f"Received query from {sender}: {msg.query}")

    # **Await the API call to make it work asynchronously**
    result = await check_information(msg.query)
    if "Not Found" in result:
        result = await check_information(msg.query, check_openai=True)

    # Respond back with classification (misinformation/information)
    if "misinformation" in result.lower():
        await ctx.send(sender, responseDeepTruthAgent(result="misinformation", source=result))
    elif "information" in result.lower():
        await ctx.send(sender, responseDeepTruthAgent(result="information", source=result))
    else:
        await ctx.send(sender, responseDeepTruthAgent(result="unable to classify", source=result))

# Collect responses from all agents
@mainAgent.on_message(model=responseDeepTruthAgent)
async def collect_responses(ctx: Context, sender: str, msg: responseDeepTruthAgent):
    global received_responses
    ctx.logger.info(f"Received response from {sender}")

    received_responses.append(msg.result)

    # Check if all agents have responded
    final_response = json.dumps(received_responses)
    ctx.logger.info(f"Final aggregated response: {final_response}")

    # Send the final response back to the sender (initial agent)
    await ctx.send(requestDeepTruthAgent.sender_address, final_response)

if __name__ == '__main__':
    mainAgent.run()
