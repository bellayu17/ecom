import requests
from flask import Flask, request, render_template, jsonify
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import re

app = Flask(__name__)

# Existing AI Functions

def get_google_trends(category):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([category], cat=0, timeframe='today 1-m', geo='', gprop='')
    trends = pytrends.related_queries()[category]['top']
    return trends.head(10).to_dict('records') if trends is not None else []

def get_ebay_trends(category, api_key):
    url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    params = {
        "q": category,
        "sort": "BEST_SELLING",
        "limit": 10
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        items = response.json().get('itemSummaries', [])
        return [{"title": item["title"], "description": item["shortDescription"]} for item in items]
    else:
        return []

def get_aliexpress_trends(category, api_key):
    url = "https://api.aliexpress.com/api/item_search"
    params = {
        "category": category,
        "sort": "SALE_DESC",
        "limit": 10,
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        items = response.json().get('items', [])
        return [{"title": item["title"], "description": item["description"]} for item in items]
    else:
        return []

def extract_keywords(description):
    words = re.findall(r'\w+', description.lower())
    keywords = {word for word in words if len(word) > 3}  # Filter out short words
    return keywords

def analyze_sentiment(description):
    sentiment_analyzer = pipeline("sentiment-analysis")
    results = sentiment_analyzer(description)
    return results

def improve_description(description):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
   
    inputs = tokenizer.encode(description, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
   
    improved_description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return improved_description

def analyze_top_products(trends_data):
    top_products = []
    for trend in trends_data['google']:
        top_products.append(trend)
   
    return top_products[:10]

def evaluate_product_trend(product, trends_data):
    trend_score = 0
    for trend in trends_data['google']:
        if product.lower() in trend.lower():
            trend_score += 1
    return trend_score

def marketing_recommendations(product, trend_score):
    recommendations = ""
    if trend_score > 5:
        recommendations = "Your product is trending well. Focus on increasing visibility on social media."
    else:
        recommendations = "Consider rebranding or enhancing the product's features to increase its market appeal."
    return recommendations

def evaluate_ecommerce_platform(product, trend_score):
    platforms = {
        "Shopee": {"focus": "budget-friendly, local market"},
        "Temu": {"focus": "innovative, unique products"},
        "Amazon": {"focus": "mass market, wide reach"},
        "Shopify": {"focus": "customized stores, niche markets"}
    }

    product_characteristics = {
        "budget-friendly": ["cheap", "affordable", "low-cost"],
        "innovative": ["new", "unique", "innovative"],
        "mass market": ["popular", "well-known", "mainstream"],
        "customized": ["custom", "niche", "specialized"]
    }

    scores = {platform: 0 for platform in platforms}

    for characteristic, keywords in product_characteristics.items():
        for keyword in keywords:
            if keyword in product.lower():
                for platform, details in platforms.items():
                    if characteristic in details["focus"]:
                        scores[platform] += 1

    recommended_platform = max(scores, key=scores.get)
    return recommended_platform, scores

# Website Analysis Functions

def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.get_text()
        return content
    else:
        return None

def generate_report(sentiment_results):
    positive = sum(1 for result in sentiment_results if result['label'] == 'POSITIVE')
    negative = sum(1 for result in sentiment_results if result['label'] == 'NEGATIVE')
    neutral = len(sentiment_results) - (positive + negative)
   
    report = {
        "total": len(sentiment_results),
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "advice": "To improve, focus on reducing negative sentiments by addressing common complaints."
    }
    return report

def evaluate_and_improve_website(url):
    content = scrape_website(url)
    if content:
        sentiment_results = analyze_sentiment(content)
        report = generate_report(sentiment_results)
        improved_description = improve_description(content)
       
        return {
            "report": report,
            "improved_description": improved_description
        }
    else:
        return {"error": "Failed to scrape the website."}

# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    category = request.form['category']
    keywords = request.form['keywords']
    description = request.form['description']
    url = request.form['url']

    # Use the existing functions to process the inputs
    ebay_api_key = "your_ebay_api_key"
    aliexpress_api_key = "your_aliexpress_api_key"

    google_trends = get_google_trends(category)
    ebay_trends = get_ebay_trends(category, ebay_api_key)
    aliexpress_trends = get_aliexpress_trends(category, aliexpress_api_key)

    trends_data = {
        "google": google_trends,
        "ebay": ebay_trends,
        "aliexpress": aliexpress_trends
    }

    top_products = analyze_top_products(trends_data)
    trend_score = evaluate_product_trend(keywords, trends_data)
    recommendations = marketing_recommendations(keywords, trend_score)
    recommended_platform, platform_scores = evaluate_ecommerce_platform(keywords, trend_score)
    improved_description = improve_description(description)

    return jsonify({
        "top_products": top_products,
        "trend_score": trend_score,
        "recommendations": recommendations,
        "recommended_platform": recommended_platform,
        "platform_scores": platform_scores,
        "improved_description": improved_description
    })

@app.route('/analyze_website', methods=['POST'])
def analyze_website():
    url = request.form['website_url']
    result = evaluate_and_improve_website(url)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)