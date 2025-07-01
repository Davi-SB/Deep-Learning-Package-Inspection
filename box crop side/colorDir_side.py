import cv2
import numpy as np
import os

def crop_box_and_return_image(image_path):
    """
    Processa uma única imagem e retorna o objeto da imagem recortada, ou None se falhar.
    """
    # ===================================================================
    # --- PARÂMETROS FINAIS (Conforme definido anteriormente) ---
    # ===================================================================
    MARGEM_X = 15
    MARGEM_Y = 15
    BLUR_KERNEL = 5
    CONTRASTE_ALPHA = 1.2
    BRILHO_BETA = -50
    SATURACAO_FACTOR = 2.4
    MORPH_KERNEL = 7
    MORPH_ITERATIONS = 0
    LOWER_CYAN = np.array([48, 0, 100])
    UPPER_CYAN = np.array([104, 255, 255])
    LOWER_WHITE = np.array([0, 0, 160])
    UPPER_WHITE = np.array([176, 60, 255])
    # ===================================================================

    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERRO] Não foi possível carregar a imagem: {image_path}")
        return None

    img_h, img_w = image.shape[:2]

    # --- Pipeline de Processamento ---
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
        print(f"  [AVISO] Nenhum contorno encontrado para: {os.path.basename(image_path)}")
        return None
        
    final_contour = max(contours, key=cv2.contourArea)

    # --- Calcular Bounding Box e Adicionar Margem ---
    x, y, w, h = cv2.boundingRect(final_contour)
    x_start = max(0, x - MARGEM_X)
    y_start = max(0, y - MARGEM_Y)
    x_end = min(img_w, x + w + MARGEM_X)
    y_end = min(img_h, y + h + MARGEM_Y)
    
    # Retorna a imagem recortada
    return image[y_start:y_end, x_start:x_end]


# =======================================================================
# --- SCRIPT PRINCIPAL DE PROCESSAMENTO EM LOTE ---
# =======================================================================

# --- IMPORTANTE: EDITE ESTA LISTA COM SEUS DIRETÓRIOS ---
# Use o formato de caminho do seu sistema operacional.
# Exemplo para Windows: r"C:\Users\SeuUsuario\Desktop\Lote1"
# Exemplo para Linux/Mac: "/home/seuusuario/documentos/lote1"
lista_de_diretorios = [
    r"damaged side",
    r"intact side",
    # Adicione quantos diretórios quiser
]

print("--- Iniciando Processamento em Lote ---")

# Loop através de cada diretório na lista
for input_dir in lista_de_diretorios:
    if not os.path.isdir(input_dir):
        print(f"\n[ERRO] Diretório não encontrado, pulando: {input_dir}")
        continue
        
    print(f"\nProcessando diretório: {input_dir}")
    
    # Cria o nome do diretório de saída
    output_dir = input_dir.rstrip('/\\') + "_cropped"
    
    # Cria o diretório de saída se ele não existir
    os.makedirs(output_dir, exist_ok=True)
    print(f"  -> Salvando em: {output_dir}")
    
    # Loop através de cada arquivo no diretório de entrada
    for filename in os.listdir(input_dir):
        # Verifica se o arquivo tem uma extensão de imagem válida
        if filename.lower().endswith('.png'):
            # Monta o caminho completo para o arquivo de entrada e saída
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Chama a função de recorte
            cropped_image = crop_box_and_return_image(image_path)
            
            # Se a função retornou uma imagem, salva no disco
            if cropped_image is not None and cropped_image.size > 0:
                cv2.imwrite(output_path, cropped_image)
                print(f"    - Imagem '{filename}' processada e salva.")
            else:
                print(f"    - Imagem '{filename}' pulada (não foi possível processar).")

print("\n--- Processamento em Lote Concluído ---")