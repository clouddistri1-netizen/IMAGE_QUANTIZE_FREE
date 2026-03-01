from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

def process_image(image_file, num_colors):
    # Abrir a imagem
    img = Image.open(image_file)
    
    # Converter para RGB para garantir consistência (remove transparência se houver)
    img = img.convert("RGB")
    
    # A magia: quantize reduz a paleta de cores
    # method=1 (MaximumCoverage) ou method=2 (FastOctree) dão bons resultados
    quantized_img = img.quantize(colors=num_colors, method=Image.Quantize.MAXCOVERAGE)
    
    # Converter de volta para RGB para poder salvar/exibir corretamente
    final_img = quantized_img.convert("RGB")
    
    # Salvar em memória (buffer)
    buffer = io.BytesIO()
    final_img.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Codificar para Base64 para enviar ao frontend
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quantize', methods=['POST'])
def quantize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    colors = int(request.form.get('colors', 2))
    
    # Validar range de cores (entre 1 e 256)
    colors = max(1, min(colors, 256))
    
    try:
        processed_base64 = process_image(file, colors)
        return jsonify({'image': processed_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)