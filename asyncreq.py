import asyncio
import aiohttp
from bs4 import BeautifulSoup

async def parse_page(session, url, payload):
    try:
        # For text payload, use 'data' with proper headers
        headers = {'Content-Type': 'application/json'}
        
        # If payload is string, we should properly format it
        # If it's JSON data, use json parameter instead
        async with session.post(
            url, 
            # data=payload,  # For plain text
            json=payload,  # Uncomment this and comment the line above if payload should be JSON
            headers=headers
        ) as response:
            html = await response.text()
            return {'url': url, 'response': html}
    except Exception as e:
        return {'url': url, 'error': str(e)}

async def main():
    # Generate a list of URLs (all pointing to the same endpoint)
    urls = [f'https://g1p3gxb3-5000.inc1.devtunnels.ms/ai-detection' for i in range(1, 101)]
    
    # Configure connection pool with limits
    conn = aiohttp.TCPConnector(limit=100)  # Allow up to 100 connections
    timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
    
    payload = {"text":"Scarcity refers to the fundamental economic problem arising from the limitation of resources in relation to unlimited wants. In the context of business activity, scarcity compels individuals, organizations, and governments to make choices about how to allocate their limited resources most effectively. This means that every choice involves an opportunity cost, which is the next best alternative forgone when a decision is made. \nFor instance, a manufacturer must decide whether to invest in new machinery or in employee training. If funds are directed towards acquiring machinery, the opportunity cost is represented by the potential enhancement of workforce skills that the training might have achieved. Scarcity thus necessitates prioritizing certain needs over others. \nFurthermore, it drives innovation and efficiency, leading firms to optimize their operations to enhance productivity and manage costs. Businesses often adopt strategies such as specialization and division of labor to maximize their output from scarce resources. Ultimately, the acknowledgement of scarcity affects not only operational decision-making within businesses but also broader economic policies and strategies aimed at sustainable growth."}
    
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = [parse_page(session, url, payload) for url in urls]
        results = await asyncio.gather(*tasks)
        print(results)
    
    return results

if __name__ == "__main__":
    results = asyncio.run(main())
    print(f"Completed {len(results)} requests")
    
    # Count successful requests vs errors
    successes = sum(1 for r in results if 'error' not in r)
    errors = sum(1 for r in results if 'error' in r)
    
    print(f"Successful: {successes}, Errors: {errors}")