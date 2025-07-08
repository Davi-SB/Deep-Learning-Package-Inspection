import cv2
import numpy as np
import os
from rembg import new_session, remove

# --- ESTÁGIO 1: FUNÇÃO DE PRÉ-RECORTE (A SUA NOVA BASE) ---
def crop_box_with_new_params(image_path: str):
   
    
    # --- Parâmetros de Margem ---
    MARGEM_X = 10
    MARGEM_Y = 10

    # --- Parâmetros de Pré-processamento e Pós-processamento (Nova Base) ---
    BLUR_KERNEL = 145
    CONTRASTE_ALPHA = 1.2
    BRILHO_BETA = -30
    SATURACAO_FACTOR = 1.0
    MORPH_KERNEL = 53
    MORPH_ITERATIONS = 2
    
    # --- Parâmetros de Cor (HSV - Nova Base) ---
    LOWER_CYAN = np.array([0, 75, 0])
    UPPER_CYAN = np.array([179, 95, 255])
    LOWER_WHITE = np.array([0, 0, 160])
    UPPER_WHITE = np.array([179, 240, 255])
    
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"    [ERRO] Não foi possível carregar a imagem: {image_path}")
        return None

    img_h, img_w = image.shape[:2]

    # --- Pipeline de Processamento para encontrar o contorno ---
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
    
    # --- Encontrar e Recortar o Contorno ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        
        return None
        
    final_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(final_contour)
    x_start = max(0, x - MARGEM_X)
    y_start = max(0, y - MARGEM_Y)
    x_end = min(img_w, x + w + MARGEM_X)
    y_end = min(img_h, y + h + MARGEM_Y)
    
    print(f"    -> Pré-recorte concluído.")
    return image[y_start:y_end, x_start:x_end]


# --- ESTÁGIO 2: FUNÇÃO DE REMOÇÃO DE FUNDO (V25) ---
def remove_background_v25(
    input_image: np.ndarray,
    output_filename: str,
    padding: int = 30
):
    
    if input_image is None or input_image.size == 0:
        print(f"    [ERRO] Dados de imagem inválidos recebidos do estágio 1.")
        return

    h_orig, w_orig = input_image.shape[:2]
    session = new_session("isnet-general-use")

    # --- Motor de Detecção (Passos 1-4) ---
    mask_pass1 = remove(input_image, session=session, only_mask=True)
    lab_clahe = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    l_clahe, a_clahe, b_clahe = cv2.split(lab_clahe)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
    cl = clahe.apply(l_clahe)
    merged_lab_clahe = cv2.merge([cl, a_clahe, b_clahe])
    enhanced_image_clahe = cv2.cvtColor(merged_lab_clahe, cv2.COLOR_LAB2BGR)
    mask_pass2 = remove(enhanced_image_clahe, session=session, only_mask=True)
    lab_bright = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    l_bright, a_bright, b_bright = cv2.split(lab_bright)
    l_boosted = np.where(cv2.inRange(l_bright, 180, 255) == 255, 255, l_bright)
    merged_lab_bright = cv2.merge([l_boosted, a_bright, b_bright])
    enhanced_image_bright = cv2.cvtColor(merged_lab_bright, cv2.COLOR_LAB2BGR)
    mask_pass3 = remove(enhanced_image_bright, session=session, only_mask=True)
    hls = cv2.cvtColor(input_image, cv2.COLOR_BGR2HLS)
    mask_pass4 = cv2.inRange(hls, np.array([0, 180, 0]), np.array([255, 255, 40]))

    # --- Combinação e Calibração Final (Passo 5) ---
    combined_mask_ia = cv2.bitwise_or(cv2.bitwise_or(mask_pass1, mask_pass2), mask_pass3)
    final_combined_mask = cv2.bitwise_or(combined_mask_ia, mask_pass4)
    dilated_mask = cv2.dilate(final_combined_mask, np.ones((7, 7), np.uint8), iterations=4)
    
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"    [ERRO] Nenhum contorno encontrado após dilatação.")
        return

    mask_reconstruida = np.zeros((h_orig, w_orig), dtype=np.uint8)
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
    for contour in significant_contours:
        cv2.drawContours(mask_reconstruida, [contour], -1, 255, thickness=cv2.FILLED)
    
    eroded_mask = cv2.erode(mask_reconstruida, np.ones((5, 5), np.uint8), iterations=7)
    mask_final = cv2.dilate(eroded_mask, np.ones((3, 3), np.uint8), iterations=6)
    
    # --- Aplicar Máscara e Salvar ---
    b, g, r = cv2.split(input_image)
    final_rgba_image = cv2.merge([b, g, r, mask_final])

    final_contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not final_contours:
        final_cropped_image = final_rgba_image
    else:
        x_coords, y_coords = [], []
        for contour in final_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])

        x_start = max(0, min(x_coords) - padding)
        y_start = max(0, min(y_coords) - padding)
        x_end = min(w_orig, max(x_coords) + padding)
        y_end = min(h_orig, max(y_coords) + padding)
        final_cropped_image = final_rgba_image[y_start:y_end, x_start:x_end]
    
    cv2.imwrite(output_filename, final_cropped_image)
    print(f"    -> Sucesso! Resultado final salvo.")


# --- EXECUÇÃO PRINCIPAL EM LOTE ---
if __name__ == "__main__":
    
    lista_de_diretorios = [
        r"damaged top_cropped",
        r"intact top_cropped",
    ]

    print("--- INICIANDO PROCESSAMENTO EM DOIS ESTÁGIOS (NOVO CROP + REMOVE BG V25) ---")

    for input_dir in lista_de_diretorios:
        if not os.path.isdir(input_dir):
            print(f"\n[ERRO] Diretório não encontrado, pulando: '{input_dir}'")
            continue
            
        print(f"\n---> Processando diretório: '{input_dir}'")
        
        # Cria o nome do diretório de saída para os resultados finais
        output_dir = input_dir.rstrip('/\\') + "_and_no_bg"
        os.makedirs(output_dir, exist_ok=True)
        print(f"  -> Salvando resultados em: '{output_dir}'")
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                print(f"\n  Processando '{filename}'...")
                image_path = os.path.join(input_dir, filename)
                base_name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_dir, base_name + '.png')
                
                try:
                    # ESTÁGIO 1: Executa o pré-recorte com os novos parâmetros
                    cropped_image_data = crop_box_with_new_params(image_path)
                    
                    # ESTÁGIO 2: Se o recorte funcionou, remove o fundo
                    if cropped_image_data is not None and cropped_image_data.size > 0:
                        remove_background_v25(
                            input_image=cropped_image_data, 
                            output_filename=output_path,
                            padding=40 # Padding final para a sombra
                        )
                    else:
                        print(f"    [AVISO] Processamento de '{filename}' pulado porque o estágio 1 falhou.")

                except Exception as e:
                    print(f"    [ERRO INESPERADO] Ocorreu um erro ao processar '{filename}': {e}")

    print("\n--- PROCESSAMENTO CONCLUÍDO ---")