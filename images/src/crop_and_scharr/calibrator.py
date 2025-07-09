import cv2
import numpy as np

def do_nothing(x):
    """Função vazia, necessária para a criação do trackbar."""
    pass

def main():
    image_path = "images\src\crop_and_scharr\caixa.png" 

    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Erro: Não foi possível carregar a imagem em '{image_path}'")
        original_image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(original_image, "Imagem nao encontrada", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Redimensiona para um tamanho máximo de exibição
    display_h, display_w = 400, 500
    h, w = original_image.shape[:2]
    if h > display_h:
        scale = display_h / h
        original_image = cv2.resize(original_image, (int(w * scale), int(h * scale)))

    # --- Cria a janela de controle e os sliders ---
    window_name = "Calibrador Sobel/Scharr"
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 600, 200)
    
    cv2.createTrackbar("Blur ksize", window_name, 1, 10, do_nothing)
    cv2.createTrackbar("Sobel ksize", window_name, 1, 3, do_nothing) # ksize deve ser 1, 3, 5, 7
    cv2.createTrackbar("Filtro (0=Sobel|1=Scharr)", window_name, 0, 1, do_nothing)

    print("\n--- Ferramenta de Calibracao de Filtros de Gradiente ---")
    print("Ajuste os sliders para visualizar os gradientes da imagem.")
    print("Pressione 'q' para sair.")
    print("-" * 60)

    while True:
        # --- Lê os valores dos sliders ---
        blur_k_raw = cv2.getTrackbarPos("Blur ksize", window_name)
        sobel_k_raw = cv2.getTrackbarPos("Sobel ksize", window_name)
        filter_type = cv2.getTrackbarPos("Filtro (0=Sobel|1=Scharr)", window_name)
        
        # Converte para tamanhos de kernel ímpares
        blur_ksize = (blur_k_raw * 2) + 1
        sobel_ksize = (sobel_k_raw * 2) + 1
        
        # Se o filtro Scharr for selecionado, o ksize é -1 por convenção do OpenCV
        if filter_type == 1:
            sobel_ksize = -1

        # --- Aplica os filtros ---
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

        # Calcula os gradientes X e Y. Usamos CV_16S para capturar valores negativos.
        grad_x = cv2.Sobel(blurred_image, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=sobel_ksize)
        grad_y = cv2.Sobel(blurred_image, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=sobel_ksize)
        
        # Converte de volta para uint8 para exibição
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # Calcula a magnitude total do gradiente
        grad_magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # --- Prepara a grade de exibição 2x2 ---
        h, w = original_image.shape[:2]
        display_grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # Converte imagens em escala de cinza para BGR para poder combiná-las
        grad_x_bgr = cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2BGR)
        grad_y_bgr = cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2BGR)
        grad_mag_bgr = cv2.cvtColor(grad_magnitude, cv2.COLOR_GRAY2BGR)
        
        # Adiciona rótulos a cada painel
        cv2.putText(original_image, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(grad_x_bgr, "Gradiente X", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(grad_y_bgr, "Gradiente Y", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(grad_mag_bgr, "Magnitude Total", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Monta a grade 2x2
        top_row = np.hstack((original_image, grad_x_bgr))
        bottom_row = np.hstack((grad_y_bgr, grad_mag_bgr))
        display_grid = np.vstack((top_row, bottom_row))

        cv2.imshow("Resultados do Filtro de Gradiente", display_grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\nCalibrador fechado.")

if __name__ == '__main__':
    main()