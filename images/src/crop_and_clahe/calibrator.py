import cv2
import numpy as np

def do_nothing(x):
    """Função vazia, necessária para a criação do trackbar."""
    pass

def main():
    image_path = "images\src\crop_and_clahe\caixa.png" 

    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Erro: Não foi possível carregar a imagem em '{image_path}'")
        print("Por favor, verifique se o nome e o caminho do arquivo estão corretos.")
        original_image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(original_image, "Imagem nao encontrada", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Redimensiona para visualização se for muito grande
    h, w = original_image.shape[:2]
    if h > 800:
        scale = 800 / h
        original_image = cv2.resize(original_image, (int(w * scale), int(h * scale)))

    # --- Cria a janela de controle e os sliders ---
    window_name = "Calibrador CLAHE"
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 600, 150)
    
    # Slider para o clipLimit (x10 para simular casas decimais)
    # Range 1 a 100 -> 0.1 a 10.0. Valor inicial 20 -> 2.0
    cv2.createTrackbar("Clip Limit (x10)", window_name, 20, 100, do_nothing)
    
    # Slider para o tileGridSize (de 1x1 a 32x32)
    cv2.createTrackbar("Tile Size", window_name, 8, 32, do_nothing)

    print("\n--- Ferramenta de Calibracao de Filtro CLAHE ---")
    print("Ajuste os sliders para equalizar o histograma e melhorar o contraste local.")
    print("Pressione 'q' para sair.")
    print("-" * 65)

    while True:
        # --- Lê os valores dos sliders ---
        clip_limit_raw = cv2.getTrackbarPos("Clip Limit (x10)", window_name)
        tile_size_raw = cv2.getTrackbarPos("Tile Size", window_name)
        
        # Converte os valores para os formatos corretos
        clip_limit = clip_limit_raw / 10.0
        # Garante que o tile size não seja 0
        tile_size = max(1, tile_size_raw)

        # --- Aplica o filtro CLAHE ---
        # 1. Converte a imagem para o espaço de cor LAB
        lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
        
        # 2. Separa os canais L, a, b
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # 3. Cria o objeto CLAHE com os parâmetros dos sliders
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        # 4. Aplica o CLAHE APENAS no canal L (Luminosidade)
        l_clahe = clahe.apply(l_channel)

        # 5. Junta o novo canal L com os canais a e b originais
        lab_clahe_image = cv2.merge((l_clahe, a_channel, b_channel))

        # 6. Converte a imagem de volta para o espaço BGR para exibição
        result_image = cv2.cvtColor(lab_clahe_image, cv2.COLOR_LAB2BGR)
        
        # --- Prepara a imagem de exibição ---
        # Adiciona texto informativo na imagem de resultado
        text_clip = f"Clip Limit: {clip_limit:.1f}"
        text_tile = f"Tile Size: {tile_size}x{tile_size}"
        cv2.putText(result_image, text_clip, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_image, text_tile, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Empilha as imagens lado a lado
        combined_display = np.hstack((original_image, result_image))
        
        cv2.imshow("Original vs. Resultado CLAHE", combined_display)

        # Espera por uma tecla. Se for 'q', sai do loop.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\nCalibrador fechado.")

if __name__ == '__main__':
    main()