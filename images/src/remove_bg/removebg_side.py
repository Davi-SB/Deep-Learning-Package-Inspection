import cv2
import numpy as np
import os
from rembg import new_session, remove

# --- FUNÇÃO DE REMOÇÃO DE FUNDO (V25) ---
def remove_background(
    image_path: str,
    output_filename: str,
    padding: int = 30
):
    """
    Carrega uma imagem de um caminho,
    aplica o processo refinado de remoção de fundo e salva o resultado.
    """
    # Carrega a imagem a partir do caminho
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"    [ERRO] Não foi possível carregar a imagem: {image_path}")
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
    print(f"  -> Sucesso! Resultado final salvo.")


# --- EXECUÇÃO PRINCIPAL EM LOTE ---
if __name__ == "__main__":
    
    # **Atenção**: Aponte esta lista para os seus diretórios de origem (os não recortados)
    lista_de_diretorios = [
        r"damaged side_cropped",
        r"intact side_cropped",
        # Adicione outros diretórios aqui se precisar
    ]

    for input_dir in lista_de_diretorios:
        if not os.path.isdir(input_dir):
            print(f"\n[ERRO] Diretório não encontrado, pulando: '{input_dir}'")
            continue

        print(f"\n---> Processando diretório: '{input_dir}'")

        # Cria o nome do diretório de saída para os resultados finais
        output_dir = input_dir.rstrip('/\\') + "_no_bg"
        os.makedirs(output_dir, exist_ok=True)
        print(f" -> Salvando resultados em: '{output_dir}'")

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                print(f"\n Processando '{filename}'...")
                image_path = os.path.join(input_dir, filename)
                base_name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_dir, base_name + '.png')

                try:
                    # Chama diretamente a função de remoção de fundo
                    remove_background(
                        image_path=image_path,
                        output_filename=output_path,
                        padding=40 # Padding final para a sombra
                    )

                except Exception as e:
                    print(f"  [ERRO INESPERADO] Ocorreu um erro ao processar '{filename}': {e}")

    print("\n--- PROCESSAMENTO CONCLUÍDO ---")