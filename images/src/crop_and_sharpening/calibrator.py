import cv2
import numpy as np

def do_nothing(x):
    """Função vazia, necessária para a criação do trackbar."""
    pass

def main():
    """
    Função principal para executar o calibrador de nitidez.
    """
    image_path = "images\src\crop_and_sharpening\caixa.png" 

    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Erro: Não foi possível carregar a imagem em '{image_path}'")
        print("Por favor, verifique se o nome e o caminho do arquivo estão corretos.")
        # Cria uma imagem preta de placeholder se a imagem não for encontrada
        original_image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(original_image, "Imagem nao encontrada", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Redimensiona para visualização se for muito grande
    h, w = original_image.shape[:2]
    if h > 800:
        scale = 800 / h
        original_image = cv2.resize(original_image, (int(w * scale), int(h * scale)))

    # --- Cria a janela de controle e o slider ---
    window_name = "Calibrador de Nitidez (Sharpening)"
    cv2.namedWindow(window_name)
    
    # O valor 5 é o padrão. Um range até 30 permite efeitos bem fortes.
    trackbar_name = "Nitidez (Centro)"
    cv2.createTrackbar(trackbar_name, window_name, 5, 30, do_nothing)

    print("\n--- Ferramenta de Calibracao de Nitidez ---")
    print("Ajuste o slider para alterar a intensidade do filtro.")
    print("Pressione 'q' para sair.")
    print("-" * 45)

    while True:
        # --- Lê o valor atual do slider ---
        center_weight = cv2.getTrackbarPos(trackbar_name, window_name)
        
        # O valor 4 produz uma imagem sem nitidez (essencialmente um filtro passa-baixa)
        # O valor 5 é a nitidez padrão.
        if center_weight < 4:
            center_weight = 4 # Evita inversão de imagem

        # --- Cria o kernel de nitidez dinamicamente ---
        sharpening_kernel = np.array([
            [-1, -1, -1],
            [-1, center_weight+0.5, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        print(center_weight+0.5)
        # --- Aplica o filtro de convolução ---
        # O ddepth -1 significa que a imagem de saída terá a mesma profundidade de bits da original.
        sharpened_image = cv2.filter2D(original_image, -1, sharpening_kernel)
        
        # --- Prepara a imagem de exibição ---
        # Adiciona texto na imagem para mostrar o valor atual
        result_display = sharpened_image.copy()
        text = f"Centro do Kernel: {center_weight}"
        cv2.putText(result_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Empilha as imagens original e com filtro lado a lado para fácil comparação
        combined_display = np.hstack((original_image, result_display))
        
        cv2.imshow(window_name, combined_display)

        # Espera por uma tecla. Se for 'q', sai do loop.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\nCalibrador fechado.")

if __name__ == '__main__':
    main()