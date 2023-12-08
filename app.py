from flask import Flask,render_template,request,jsonify

from chat import get_response

app=Flask(__name__)
# Initialize an empty list to store chat messages
chat_history = []

@app.route('/store_message', methods=['POST'])
def store_message():
    content = request.json['content']
    chat_history.append(content)
    return jsonify({"message": "Message stored successfully"})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)
@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    #TODO:check if text is valid
    response=get_response(text)
    message={"answer":response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
