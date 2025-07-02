import cv2
import numpy as np

def crop_box_with_margin(image_path):
    """
    Realiza o recorte da caixa usando os parâmetros de cor calibrados e
    adiciona uma margem de segurança ao redor do recorte final.
    """
    # ===================================================================
    # --- PARÂMETROS FINAIS (Transcribed from user image) ---
    # ===================================================================
    
    # --- Parâmetros da Margem de Erro (em pixels) ---
    MARGEM_X = 10  # Adiciona pixels na esquerda e na direita
    MARGEM_Y = 10  # Adiciona pixels em cima e embaixo

    # --- Parâmetros de Pré-processamento e Pós-processamento ---
    BLUR_KERNEL = 145
    CONTRASTE_ALPHA = 1.2  # Valor 12 / 10.0
    BRILHO_BETA = -30      # Valor 20 - 50
    SATURACAO_FACTOR = 1.0 # Valor 10 / 10.0
    MORPH_KERNEL = 53
    MORPH_ITERATIONS = 2
    
    # --- Parâmetros de Cor (HSV) ---
    LOWER_CYAN = np.array([0, 75, 0])
    UPPER_CYAN = np.array([179, 95, 255])
    
    # ATENÇÃO: V_min_W e V_max_W não estavam na imagem. Foram usados valores padrão.
    LOWER_WHITE = np.array([0, 0, 160])
    UPPER_WHITE = np.array([179, 240, 255])

    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem em '{image_path}'")
        return

    img_h, img_w = image.shape[:2]

    # --- 1. Pipeline de Processamento de Imagem ---
    processed_image = image.copy()
    if BLUR_KERNEL > 1:
        processed_image = cv2.GaussianBlur(processed_image, (BLUR_KERNEL, BLUR_KERNEL), 0)
    
    processed_image = cv2.convertScaleAbs(processed_image, alpha=CONTRASTE_ALPHA, beta=BRILHO_BETA)
    
    hsv_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = np.clip(s * SATURACAO_FACTOR, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge([h, s, v])
    
    mask_cyan = cv2.inRange(final_hsv, LOWER_CYAN, UPPER_CYAN)
    mask_white = cv2.inRange(final_hsv, LOWER_WHITE, UPPER_WHITE)
    final_mask = cv2.bitwise_or(mask_cyan, mask_white)

    if MORPH_KERNEL > 1 and MORPH_ITERATIONS > 0:
        kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    
    # --- 2. Encontrar o Contorno da Caixa ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Nenhum contorno encontrado com os parâmetros fornecidos.")
        return
        
    final_contour = max(contours, key=cv2.contourArea)

    # --- 3. Calcular Bounding Box e Adicionar Margem ---
    # Usamos um retângulo simples (não rotacionado) para facilitar a adição da margem
    x, y, w, h = cv2.boundingRect(final_contour)
    
    # Adiciona a margem e garante que as coordenadas não saiam da imagem
    x_start = max(0, x - MARGEM_X)
    y_start = max(0, y - MARGEM_Y)
    x_end = min(img_w, x + w + MARGEM_X)
    y_end = min(img_h, y + h + MARGEM_Y)
    
    # --- 4. Recortar a Imagem Original ---
    cropped_box = image[y_start:y_end, x_start:x_end]
    
    # Salva o resultado
    cv2.imwrite("box crop top/output_final_cropped_box.png", cropped_box)
    
    # Opcional: Salva uma imagem de depuração para ver a área de recorte
    debug_image = image.copy()
    cv2.rectangle(debug_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.imwrite("box crop top/output_debug_crop_area.png", debug_image)

    print("Processo concluído!")
    print(f"Caixa recortada e salva como 'output_final_cropped_box.png'")
    print(f"Área de recorte com margem salva como 'output_debug_crop_area.png'")


# --- Execute o script com a imagem desejada ---
# Substitua pelo nome do arquivo que você quer processar.
crop_box_with_margin("box crop top/caixa.png")