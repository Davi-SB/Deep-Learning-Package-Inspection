import cv2
import numpy as np
import os

def apply_scharr_filter(input_path: str, output_path: str):
    try:
        # Define os parâmetros perfeitos que você encontrou no calibrador
        blur_ksize = 1 # Valor 0 no slider -> (0*2)+1 = 1
        # O filtro selecionado foi o Scharr (valor 1 no slider), que usa ksize = -1
        scharr_ksize = -1 

        # Lê a imagem de entrada
        image = cv2.imread(input_path)
        if image is None:
            print(f"  [ERRO] Não foi possível ler o arquivo: {input_path}")
            return False

        # --- Aplica a sequência de filtros ---
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Um blur com kernel 1x1 não tem efeito, mas mantemos a lógica
        if blur_ksize > 0:
            blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)
        else:
            blurred_image = gray_image

        # Calcula os gradientes X e Y usando o filtro Scharr (ksize=-1)
        grad_x = cv2.Sobel(blurred_image, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=scharr_ksize)
        grad_y = cv2.Sobel(blurred_image, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=scharr_ksize)
        
        # Converte de volta para uint8
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # Calcula a magnitude total do gradiente
        grad_magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Salva a imagem resultante
        cv2.imwrite(output_path, grad_magnitude)
        return True

    except Exception as e:
        print(f"  [ERRO] Ocorreu uma exceção ao processar '{input_path}': {e}")
        return False

def main():
    root_input_dir = "images/crop_only"
    root_output_dir = "images/crop_and_scharr"

    valid_extensions = ('.png')

    print(f"--- Iniciando Processamento em Lote (Filtro Scharr) ---")
    print(f"Diretório de Origem: {root_input_dir}")
    print(f"Diretório de Saída:  {root_output_dir}")
    print(f"Parâmetros: blur_ksize=1, filter=Scharr")
    print("-" * 60)

    if not os.path.isdir(root_input_dir):
        print(f"[FALHA] O diretório de origem não foi encontrado: '{root_input_dir}'")
        return

    processed_count = 0
    failed_count = 0

    # os.walk() varre recursivamente o diretório de cima para baixo
    for dirpath, _, filenames in os.walk(root_input_dir):
        # Cria a estrutura de pastas correspondente no diretório de saída
        relative_path = os.path.relpath(dirpath, root_input_dir)
        output_subdir = os.path.join(root_output_dir, relative_path)
        
        os.makedirs(output_subdir, exist_ok=True)
        
        for filename in filenames:
            if filename.lower().endswith(valid_extensions):
                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(output_subdir, filename)
                
                print(f"Processando: {input_path}")
                
                # Chama a função para aplicar o filtro Scharr
                success = apply_scharr_filter(input_path, output_path)
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
    
    print("-" * 60)
    print("--- Processamento em Lote Concluído ---")
    print(f"Imagens processadas com sucesso: {processed_count}")
    print(f"Falhas: {failed_count}")


if __name__ == '__main__':
    main()