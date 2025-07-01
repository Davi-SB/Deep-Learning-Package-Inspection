import cv2
import numpy as np

def do_nothing(x):
    pass

# --- Carregar Imagem ---
image_path = 'caixa.png'  # <-- COLOQUE O NOME DA SUA IMAGEM AQUI
image = cv2.imread(image_path)
h, w = image.shape[:2]
if h > 800:
    scale = 800 / h
    image = cv2.resize(image, (int(w * scale), int(h * scale)))

# --- Criar Janela de Controle e Sliders ---
cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 700, 600)

# Sliders de Pré-processamento
cv2.createTrackbar('Blur', 'Trackbars', 1, 210, do_nothing)
cv2.createTrackbar('Contraste (x10)', 'Trackbars', 10, 300, do_nothing) # 1.0 a 3.0
cv2.createTrackbar('Brilho', 'Trackbars', 50, 1000, do_nothing) # -50 a 50
cv2.createTrackbar('Saturacao (x10)', 'Trackbars', 10, 300, do_nothing) # 1.0 a 3.0

# Sliders de Pós-processamento da Máscara
cv2.createTrackbar('Kernel Morph', 'Trackbars', 5, 210, do_nothing)
cv2.createTrackbar('Iteracoes Close', 'Trackbars', 3, 100, do_nothing)

# Sliders para Cor 1 (Ciano)
cv2.createTrackbar('H_min_C', 'Trackbars', 78, 179, do_nothing)
cv2.createTrackbar('H_max_C', 'Trackbars', 95, 179, do_nothing)
cv2.createTrackbar('S_min_C', 'Trackbars', 100, 255, do_nothing)
cv2.createTrackbar('S_max_C', 'Trackbars', 255, 255, do_nothing)
cv2.createTrackbar('V_min_C', 'Trackbars', 100, 255, do_nothing)
cv2.createTrackbar('V_max_C', 'Trackbars', 255, 255, do_nothing)

# Sliders para Cor 2 (Branco)
cv2.createTrackbar('H_min_W', 'Trackbars', 0, 179, do_nothing)
cv2.createTrackbar('H_max_W', 'Trackbars', 179, 179, do_nothing)
cv2.createTrackbar('S_min_W', 'Trackbars', 0, 255, do_nothing)
cv2.createTrackbar('S_max_W', 'Trackbars', 50, 255, do_nothing)
cv2.createTrackbar('V_min_W', 'Trackbars', 160, 255, do_nothing)
cv2.createTrackbar('V_max_W', 'Trackbars', 255, 255, do_nothing)

while True:
    # --- 1. Ler valores dos Sliders ---
    blur_k = cv2.getTrackbarPos('Blur', 'Trackbars')
    alpha = cv2.getTrackbarPos('Contraste (x10)', 'Trackbars') / 10.0
    beta = cv2.getTrackbarPos('Brilho', 'Trackbars') - 50
    sat_factor = cv2.getTrackbarPos('Saturacao (x10)', 'Trackbars') / 10.0
    morph_k = cv2.getTrackbarPos('Kernel Morph', 'Trackbars')
    morph_iter = cv2.getTrackbarPos('Iteracoes Close', 'Trackbars')

    # Garante que os kernels sejam ímpares e maiores que 0
    if blur_k % 2 == 0: blur_k += 1
    if morph_k % 2 == 0: morph_k += 1

    # --- 2. Pré-processamento da Imagem ---
    processed_image = image.copy()
    
    # Aplica Blur
    if blur_k > 1:
        processed_image = cv2.GaussianBlur(processed_image, (blur_k, blur_k), 0)
    
    # Aplica Contraste e Brilho
    processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
    
    # Converte para HSV para ajustar a saturação
    hsv_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # Aplica fator de Saturação, garantindo que não ultrapasse 255
    s = np.clip(s * sat_factor, 0, 255).astype(np.uint8)
    
    # Remonta a imagem HSV
    final_hsv = cv2.merge([h, s, v])

    # --- 3. Segmentação por Cor ---
    h_min_c, h_max_c, s_min_c, s_max_c, v_min_c, v_max_c = [cv2.getTrackbarPos(n, 'Trackbars') for n in ['H_min_C', 'H_max_C', 'S_min_C', 'S_max_C', 'V_min_C', 'V_max_C']]
    h_min_w, h_max_w, s_min_w, s_max_w, v_min_w, v_max_w = [cv2.getTrackbarPos(n, 'Trackbars') for n in ['H_min_W', 'H_max_W', 'S_min_W', 'S_max_W', 'V_min_W', 'V_max_W']]
    
    mask_cyan = cv2.inRange(final_hsv, np.array([h_min_c,s_min_c,v_min_c]), np.array([h_max_c,s_max_c,v_max_c]))
    mask_white = cv2.inRange(final_hsv, np.array([h_min_w,s_min_w,v_min_w]), np.array([h_max_w,s_max_w,v_max_w]))
    combined_mask = cv2.bitwise_or(mask_cyan, mask_white)

    # --- 4. Pós-processamento da Máscara ---
    if morph_k > 1 and morph_iter > 0:
        kernel = np.ones((morph_k, morph_k), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

    # --- 5. Mostrar Resultados ---
    cv2.imshow('Original', image)
    cv2.imshow('Pre-processada', cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR))
    cv2.imshow('Mascara Final', combined_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Ao sair, imprime os últimos valores para facilitar o 'copia e cola'
print("\n--- Parâmetros Finais ---")
print(f"BLUR_KERNEL = {blur_k}")
print(f"CONTRASTE_ALPHA = {alpha}")
print(f"BRILHO_BETA = {beta}")
print(f"SATURACAO_FACTOR = {sat_factor}")
print(f"MORPH_KERNEL = {morph_k}")
print(f"MORPH_ITERATIONS = {morph_iter}")
print("-" * 25)
print(f"LOWER_CYAN = np.array([{h_min_c}, {s_min_c}, {v_min_c}])")
print(f"UPPER_CYAN = np.array([{h_max_c}, {s_max_c}, {v_max_c}])")
print(f"LOWER_WHITE = np.array([{h_min_w}, {s_min_w}, {v_min_w}])")
print(f"UPPER_WHITE = np.array([{h_max_w}, {s_max_w}, {v_max_w}])")