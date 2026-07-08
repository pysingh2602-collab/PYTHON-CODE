from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

orders = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/save_order", methods=["POST"])
def save_order():

    data = request.json

    orders.append(data)

    return jsonify({
        "message":"Order Saved Successfully"
    })

@app.route("/orders")
def get_orders():
    return jsonify(orders)

if __name__ == "__main__":
    app.run(debug=True)