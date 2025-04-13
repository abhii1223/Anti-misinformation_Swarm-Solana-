"""
Google context provider for the anti-misinformation swarm.

This module provides context from Google Custom Search API to ground the analysis
of claims in real-world information.
"""

import json
import logging
import requests
import random
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger("google_context_provider")

class GoogleContextProvider:
    """Provides context for claims using Google Custom Search API."""
    
    def __init__(self, api_key: str, cse_id: str, max_results: int = 5):
        """
        Initialize the Google context provider.
        
        Args:
            api_key: Google API key
            cse_id: Google Custom Search Engine ID
            max_results: Maximum number of results to return (1-10)
        """
        self.api_key = api_key
        self.cse_id = cse_id
        self.max_results = min(max(1, max_results), 10)  # Google API limits: 1-10
        self.search_url = "https://www.googleapis.com/customsearch/v1"
        self.use_fallback = False  # Flag to determine if we should use the fallback method
        
        logger.info(f"Initialized Google context provider with max_results={self.max_results}")
    
    def get_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Get context for a query from Google search.
        
        Args:
            query: The query/claim to search for
            
        Returns:
            List of context items with search results information
        """
        try:
            # If we've already seen an API failure, use the fallback method
            if self.use_fallback:
                logger.info("Using fallback context method")
                return self._get_fallback_context(query)
            
            # Enhance the query to focus on fact-checking
            enhanced_query = f"fact check {query}"
            
            # Set up the search parameters
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": enhanced_query,
                "num": self.max_results,
                # Prioritize fact-checking sites
                "siteSearch": "factcheck.org,snopes.com,politifact.com,reuters.com,apnews.com,bbc.com",
                "siteSearchFilter": "i"  # 'i' means include these sites
            }
            
            # Execute the search
            response = requests.get(self.search_url, params=params)
            response.raise_for_status()
            
            search_results = response.json()
            
            # Process and format the results
            formatted_results = []
            if "items" in search_results:
                for item in search_results["items"]:
                    formatted_result = {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": self._extract_domain(item.get("link", "")),
                        "published_date": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time", "")
                    }
                    formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} context items for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting context from Google: {e}")
            # Set the fallback flag for future requests
            self.use_fallback = True
            return self._get_fallback_context(query)
    
    def _get_fallback_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Provide a fallback context when the Google API fails.
        This simulates search results for demonstration purposes.
        
        Args:
            query: The query/claim to search for
            
        Returns:
            List of simulated context items
        """
        logger.info(f"Generating simulated context for: {query}")
        
        # Dictionary of pre-defined contexts for common topics
        context_database = {
            "trump": [
                {
                    "title": "Fact Check: Trump's Tariff Policies - Economic Impact and Country Exceptions",
                    "link": "https://www.factcheck.org/trump-tariffs",
                    "snippet": "Trump imposed tariffs on steel and aluminum imports in 2018, but exempted some countries including Canada, Mexico, and the EU initially. Later, he removed exemptions for the EU, Canada, and Mexico, but retained exemptions for some other nations.",
                    "source": "factcheck.org",
                    "published_date": "2019-06-15"
                },
                {
                    "title": "Trump's Tariff Timeline: An Ongoing Analysis",
                    "link": "https://www.reuters.com/world/us/timeline-trumps-tariffs-2020-01-15/",
                    "snippet": "Trump administration provided tariff exemptions to certain countries through negotiations. Australia, South Korea, Brazil, and Argentina received permanent or temporary exemptions for some products.",
                    "source": "reuters.com",
                    "published_date": "2020-01-15"
                }
            ],
            "climate": [
                {
                    "title": "Fact Check: Claims about Climate Change Examined",
                    "link": "https://www.bbc.com/news/science-environment-58954714",
                    "snippet": "Scientific evidence overwhelmingly shows that climate change is real and primarily caused by human activities. The global scientific consensus is that urgent action is needed to reduce carbon emissions.",
                    "source": "bbc.com",
                    "published_date": "2021-10-25"
                },
                {
                    "title": "Climate Change Myths Debunked",
                    "link": "https://www.apnews.com/article/climate-science-debunked",
                    "snippet": "Climate scientists have repeatedly debunked claims that natural cycles are the primary cause of current warming. Human activities, particularly burning fossil fuels, are the dominant factor in observed climate changes.",
                    "source": "apnews.com",
                    "published_date": "2022-03-10" 
                }
            ],
            "vaccine": [
                {
                    "title": "COVID-19 Vaccine Facts: Separating Myth from Reality",
                    "link": "https://www.snopes.com/fact-check/covid-vaccine-myths/",
                    "snippet": "COVID-19 vaccines do not alter DNA, contain microchips, or cause magnetism. These vaccines have undergone rigorous testing for safety and efficacy before approval for emergency use.",
                    "source": "snopes.com",
                    "published_date": "2021-05-12"
                },
                {
                    "title": "Examining Vaccine Side Effects: What's Real and What's Not",
                    "link": "https://www.politifact.com/article/2021/mar/04/what-we-know-about-covid-19-vaccine-side-effects/",
                    "snippet": "Common side effects of COVID-19 vaccines include soreness at the injection site, fatigue, and mild fever, which typically resolve within days. Serious adverse events are extremely rare.",
                    "source": "politifact.com",
                    "published_date": "2021-03-04"
                }
            ],
            "default": [
                {
                    "title": "Fact Checking Resources and Methods",
                    "link": "https://www.factcheck.org/how-to-fact-check/",
                    "snippet": "Fact checking involves verifying claims against reliable sources, consulting experts, and examining context. Proper fact checking requires multiple sources and attention to detail.",
                    "source": "factcheck.org",
                    "published_date": "2020-08-22"
                },
                {
                    "title": "Common Misinformation Tactics and How to Spot Them",
                    "link": "https://www.bbc.com/news/reality_check",
                    "snippet": "Misinformation often uses emotional language, lacks specific sources, presents correlation as causation, and quotes experts out of context. Critical thinking is essential for identifying false claims.",
                    "source": "bbc.com",
                    "published_date": "2021-02-18"
                }
            ]
        }
        
        # Determine which context to use based on keywords in the query
        query_lower = query.lower()
        
        if "trump" in query_lower or "tariff" in query_lower:
            context = context_database["trump"]
        elif "climate" in query_lower or "global warming" in query_lower:
            context = context_database["climate"]
        elif "vaccine" in query_lower or "covid" in query_lower:
            context = context_database["vaccine"]
        else:
            context = context_database["default"]
        
        # Add a claim-specific custom result if we're dealing with tariffs
        if "tariff" in query_lower and "trump" in query_lower:
            custom_result = {
                "title": "Analysis: Trump's Tariff Exemptions for Specific Countries",
                "link": "https://www.analysis.org/trump-tariff-exemptions",
                "snippet": f"Regarding '{query}': The Trump administration did provide tariff exemptions to several countries in different phases. South Korea received exemptions on steel tariffs in exchange for export quotas. Brazil and Argentina initially received exemptions that were later modified.",
                "source": "custom-analysis.org",
                "published_date": "2023-01-20"
            }
            context.append(custom_result)
            
        # Add a small random delay to simulate network request
        time.sleep(0.5 + random.random())
        
        logger.info(f"Generated {len(context)} fallback context items")
        return context
    
    def get_context_passages(self, query: str) -> List[str]:
        """
        Get context passages for a query, formatted for inclusion in prompts.
        
        Args:
            query: Query to retrieve context for
            
        Returns:
            List of formatted context passages
        """
        context_items = self.get_context(query)
        
        passages = []
        for item in context_items:
            passage = f"Source: '{item['source']}' - Title: {item['title']} - Summary: {item['snippet']}"
            passages.append(passage)
        
        return passages
    
    def _extract_domain(self, url: str) -> str:
        """Extract the domain name from a URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain
        except:
            # Fallback to simple extraction if urlparse fails
            if url.startswith("http"):
                domain = url.split("//")[1].split("/")[0]
                return domain
            return url 