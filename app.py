from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Загрузка модели
model = torch.jit.load("scripted_model.pt")
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_tensor = torch.tensor(data['input'], dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5001)