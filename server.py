from flask import Flask, request, jsonify, render_template
from query_processor import search
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_endpoint():
    query = request.args.get('q')

    start_time = time.time()
    results = search(query, top_k=5)
    end_time = time.time()

    return jsonify({
        'urls': [result[2] for result in results],
        'time': (end_time - start_time) * 1000 # milliseconds
    })

if __name__ == '__main__':
    app.run(debug=True)
