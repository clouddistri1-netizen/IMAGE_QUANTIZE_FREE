from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage import color as skcolor
import os

app = Flask(__name__)

def process_image(image_file, num_colors):
    img = Image.open(image_file)

    # ── 1. SEPARAR ALPHA ANTES DE QUALQUER COISA ─────────────────────────────
    # Pixel art frequentemente tem alpha binário (0 ou 255) ou semitransparente.
    # Misturar com fundo branco ANTES do K-Means contamina as cores de borda.
    has_alpha = img.mode in ('RGBA', 'LA') or \
                (img.mode == 'P' and 'transparency' in img.info)

    if has_alpha:
        img = img.convert('RGBA')
        img_array = np.array(img, dtype=np.float32)
        alpha_channel = img_array[:, :, 3]          # Guardar alpha original
        rgb_array     = img_array[:, :, :3] / 255.0 # RGB normalizado

        # Máscara: apenas pixels suficientemente opacos entram no K-Means
        opaque_mask = alpha_channel > 10  # threshold: ignora pixels quase invisíveis
    else:
        img = img.convert('RGB')
        img_array    = np.array(img, dtype=np.float32)
        rgb_array    = img_array / 255.0
        alpha_channel = None
        opaque_mask  = np.ones(rgb_array.shape[:2], dtype=bool)

    h, w, _ = rgb_array.shape

    # ── 2. K-MEANS APENAS NOS PIXELS OPACOS (em espaço LAB) ──────────────────
    # Pixels transparentes NÃO entram no clustering — sem contaminação de borda.
    pixels_rgb_opaque = rgb_array[opaque_mask]           # shape: (N_opaque, 3)
    pixels_lab_opaque = skcolor.rgb2lab(
        pixels_rgb_opaque.reshape(-1, 1, 3)              # skimage espera 3D
    ).reshape(-1, 3)

    # Clamp num_colors ao número de pixels únicos disponíveis
    n_opaque = len(pixels_lab_opaque)
    num_colors = min(num_colors, n_opaque)

    n_pixels = h * w
    if n_pixels > 200_000:
        kmeans = MiniBatchKMeans(
            n_clusters=num_colors, n_init=5,
            max_iter=300, batch_size=4096, random_state=42
        )
    else:
        kmeans = KMeans(
            n_clusters=num_colors, n_init=15,
            max_iter=500, random_state=42
        )

    kmeans.fit(pixels_lab_opaque)
    centers_lab = kmeans.cluster_centers_  # paleta em LAB

    # ── 3. REMAP: substituir cada pixel opaco pelo centro mais próximo ────────
    all_pixels_lab = skcolor.rgb2lab(rgb_array.reshape(-1, 1, 3)).reshape(-1, 3)

    # Calcular distâncias de todos os pixels para todos os centros
    # shape: (N_pixels, num_colors)
    diffs  = all_pixels_lab[:, np.newaxis, :] - centers_lab[np.newaxis, :, :]
    dists  = np.linalg.norm(diffs, axis=2)
    labels = np.argmin(dists, axis=1)                    # índice do centro mais próximo

    result_lab = centers_lab[labels].reshape(h, w, 3)

    # ── 4. CONVERTER DE VOLTA PARA RGB ────────────────────────────────────────
    result_rgb = skcolor.lab2rgb(result_lab)
    result_rgb = np.clip(result_rgb * 255, 0, 255).astype(np.uint8)

    # ── 5. REAPLICAR O ALPHA ORIGINAL (SEM MODIFICAÇÃO) ───────────────────────
    # O alpha volta exatamente como era — pixels transparentes ficam transparentes,
    # sem borda branca, sem halo.
    if alpha_channel is not None:
        alpha_uint8 = alpha_channel.astype(np.uint8)
        final_array = np.dstack([result_rgb, alpha_uint8])  # RGBA
        final_img   = Image.fromarray(final_array, 'RGBA')
    else:
        final_img = Image.fromarray(result_rgb, 'RGB')

    # ── 6. SERIALIZAR ─────────────────────────────────────────────────────────
    buffer = io.BytesIO()
    final_img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quantize', methods=['POST'])
def quantize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file   = request.files['image']
    colors = int(request.form.get('colors', 8))
    colors = max(2, min(colors, 256))

    try:
        processed = process_image(file, colors)
        return jsonify({'image': processed})
    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)