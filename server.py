from flask import Flask, request, jsonify
from query_processor import search

app = Flask(__name__)

@app.route('/search', methods=['GET'])
def search_endpoint():
    query = request.args.get('q')

    results = search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)


