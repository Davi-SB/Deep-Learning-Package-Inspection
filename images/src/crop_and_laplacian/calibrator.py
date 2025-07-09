import cv2
import numpy as np

def do_nothing(x):
    """Função vazia, necessária para a criação do trackbar."""
    pass

def main():
    image_path = "images\src\crop_and_laplacian\caixa.png" 

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
    window_name = "Calibrador de Filtro Laplaciano"
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 600, 150)
    
    cv2.createTrackbar("Blur ksize", window_name, 1, 10, do_nothing)
    cv2.createTrackbar("Laplacian ksize", window_name, 1, 10, do_nothing)

    print("\n--- Ferramenta de Calibracao de Filtro Laplaciano ---")
    print("Ajuste os sliders para alterar o desfoque e o tamanho do kernel.")
    print("Pressione 'q' para sair.")
    print("-" * 60)

    while True:
        # --- Lê os valores dos sliders e os converte para tamanhos de kernel ímpares ---
        blur_k_raw = cv2.getTrackbarPos("Blur ksize", window_name)
        laplacian_k_raw = cv2.getTrackbarPos("Laplacian ksize", window_name)
        
        # A fórmula (valor * 2) + 1 garante um resultado ímpar e positivo (1, 3, 5...)
        blur_ksize = (blur_k_raw * 2) + 1
        laplacian_ksize = (laplacian_k_raw * 2) + 1

        # --- Aplica os filtros ---
        # 1. Converte para escala de cinza
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Aplica desfoque para reduzir ruído
        blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

        # 3. Aplica o filtro Laplaciano
        # Usamos cv2.CV_16S para a profundidade de dados para evitar perda de informação
        # nas bordas (valores negativos seriam cortados em uint8).
        laplacian_filtered = cv2.Laplacian(blurred_image, ddepth=cv2.CV_16S, ksize=laplacian_ksize)
        
        # Converte o resultado de volta para um formato de 8-bits para exibição
        laplacian_display = cv2.convertScaleAbs(laplacian_filtered)
        
        # --- Prepara a imagem de exibição ---
        # Converte a imagem de bordas para 3 canais para poder empilhá-la com a original
        laplacian_display_bgr = cv2.cvtColor(laplacian_display, cv2.COLOR_GRAY2BGR)

        # Adiciona texto informativo na imagem de resultado
        text_blur = f"Blur ksize: {blur_ksize}"
        text_laplacian = f"Laplacian ksize: {laplacian_ksize}"
        cv2.putText(laplacian_display_bgr, text_blur, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(laplacian_display_bgr, text_laplacian, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Empilha as imagens lado a lado
        combined_display = np.hstack((original_image, laplacian_display_bgr))
        
        cv2.imshow("Original vs. Resultado Laplaciano", combined_display)

        # Espera por uma tecla. Se for 'q', sai do loop.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\nCalibrador fechado.")

if __name__ == '__main__':
    main()