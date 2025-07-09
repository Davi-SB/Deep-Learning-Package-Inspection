import cv2
import numpy as np
import os

def apply_laplacian_filter(input_path: str, output_path: str):
    try:
        blur_ksize = 3
        laplacian_ksize = 5

        # Lê a imagem de entrada
        image = cv2.imread(input_path)
        if image is None:
            print(f"  [ERRO] Não foi possível ler o arquivo: {input_path}")
            return False

        # --- Aplica a sequência de filtros ---
        # 1. Converte para escala de cinza
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Aplica desfoque com o ksize definido
        blurred_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

        # 3. Aplica o filtro Laplaciano com o ksize definido
        laplacian_filtered = cv2.Laplacian(blurred_image, ddepth=cv2.CV_16S, ksize=laplacian_ksize)
        
        # 4. Converte o resultado de volta para um formato de 8-bits para salvar
        final_image = cv2.convertScaleAbs(laplacian_filtered)
        
        # Salva a imagem resultante
        cv2.imwrite(output_path, final_image)
        return True

    except Exception as e:
        print(f"  [ERRO] Ocorreu uma exceção ao processar '{input_path}': {e}")
        return False


if __name__ == '__main__':
    root_input_dir = "images/crop_only"
    root_output_dir = "images/crop_and_laplacian"

    valid_extensions = ('.png')

    print(f"--- Iniciando Processamento em Lote (Filtro Laplaciano) ---")
    print(f"Diretório de Origem: {root_input_dir}")
    print(f"Diretório de Saída:  {root_output_dir}")
    print(f"Parâmetros: blur_ksize=3, laplacian_ksize=5")
    print("-" * 60)

    if not os.path.isdir(root_input_dir):
        print(f"[FALHA] O diretório de origem não foi encontrado: '{root_input_dir}'")
        input("Pressione Enter para sair...")
        exit(1)

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
                
                # Chama a nova função para aplicar o filtro Laplaciano
                success = apply_laplacian_filter(input_path, output_path)
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
    
    print("-" * 60)
    print("--- Processamento em Lote Concluído ---")
    print(f"Imagens processadas com sucesso: {processed_count}")
    print(f"Falhas: {failed_count}")