import cv2
import numpy as np

def do_nothing(x):
    """Função vazia, necessária para a criação do trackbar."""
    pass

def main():
    """
    Função principal para executar o calibrador do detector de bordas Canny.
    """
    image_path = "images\src\crop_and_canny\caixa.png" 

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
    window_name = "Calibrador de Detector de Bordas Canny"
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 600, 200)
    
    cv2.createTrackbar("Blur ksize", window_name, 1, 10, do_nothing)
    cv2.createTrackbar("Threshold 1", window_name, 50, 255, do_nothing)
    cv2.createTrackbar("Threshold 2", window_name, 150, 255, do_nothing)

    print("\n--- Ferramenta de Calibracao do Detector Canny ---")
    print("Ajuste os sliders para encontrar as bordas desejadas.")
    print("Pressione 'q' para sair.")
    print("-" * 55)

    while True:
        # --- Lê os valores dos sliders ---
        blur_k_raw = cv2.getTrackbarPos("Blur ksize", window_name)
        threshold1 = cv2.getTrackbarPos("Threshold 1", window_name)
        threshold2 = cv2.getTrackbarPos("Threshold 2", window_name)
        
        # Converte o valor do slider para um tamanho de kernel ímpar (1, 3, 5...)
        blur_ksize = (blur_k_raw * 2) + 1

        # --- Aplica os filtros ---
        # 1. Converte para escala de cinza
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Aplica desfoque para reduzir ruído
        blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

        # 3. Aplica o detector de bordas Canny
        canny_edges = cv2.Canny(blurred_image, threshold1, threshold2)
        
        # --- Prepara a imagem de exibição ---
        # Converte a imagem de bordas (preto e branco) para 3 canais (BGR)
        # para poder empilhá-la com a imagem original colorida.
        canny_display_bgr = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)

        # Adiciona texto informativo na imagem de resultado
        text_blur = f"Blur: {blur_ksize}"
        text_thresh = f"Thresh: {threshold1}, {threshold2}"
        cv2.putText(canny_display_bgr, text_blur, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canny_display_bgr, text_thresh, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Empilha as imagens lado a lado
        combined_display = np.hstack((original_image, canny_display_bgr))
        
        cv2.imshow("Original vs. Resultado Canny", combined_display)

        # Espera por uma tecla. Se for 'q', sai do loop.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\nCalibrador fechado.")

if __name__ == '__main__':
    main()