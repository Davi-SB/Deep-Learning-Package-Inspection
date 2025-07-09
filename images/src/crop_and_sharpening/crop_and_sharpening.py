import cv2
import numpy as np
import os

def apply_sharpening(input_path: str, output_path: str):
    try:
        # Lê a imagem de entrada
        image = cv2.imread(input_path)
        if image is None:
            print(f"  [ERRO] Não foi possível ler o arquivo: {input_path}")
            return False

        # Define o kernel de nitidez com o valor fixo de 8.5
        sharpening_kernel = np.array([
            [-1, -1, -1],
            [-1, 8.5, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

        # Aplica o filtro de convolução
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
        
        # Salva a imagem resultante
        cv2.imwrite(output_path, sharpened_image)
        return True

    except Exception as e:
        print(f"  [ERRO] Ocorreu uma exceção ao processar '{input_path}': {e}")
        return False

if __name__ == '__main__':
    # Define os diretórios raiz de entrada e saída
    root_input_dir = os.path.join("images", "crop_only")
    root_output_dir = os.path.join("images", "crop_and_sharpening")

    # Define as extensões de imagem válidas
    valid_extensions = ('.png')

    print(f"--- Iniciando Processamento em Lote ---")
    print(f"Diretório de Origem: {root_input_dir}")
    print(f"Diretório de Saída:  {root_output_dir}")
    print("-" * 45)

    if not os.path.isdir(root_input_dir):
        print(f"[FALHA] O diretório de origem não foi encontrado: '{root_input_dir}'")
        input("Pressione Enter para sair...")
        exit(1)

    processed_count = 0
    failed_count = 0

    # os.walk() varre recursivamente o diretório de cima para baixo
    for dirpath, _, filenames in os.walk(root_input_dir):
        # Cria a estrutura de pastas correspondente no diretório de saída
        # Ex: "images\crop_only\A\B" -> "images\crop_and_sharpening\A\B"
        relative_path = os.path.relpath(dirpath, root_input_dir)
        output_subdir = os.path.join(root_output_dir, relative_path)
        
        # Cria o subdiretório de saída se ele não existir
        os.makedirs(output_subdir, exist_ok=True)
        
        for filename in filenames:
            # Verifica se o arquivo tem uma extensão de imagem válida
            if filename.lower().endswith(valid_extensions):
                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(output_subdir, filename)
                
                print(f"Processando: {input_path}")
                
                # Aplica o filtro
                success = apply_sharpening(input_path, output_path)
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
    
    print("-" * 45)
    print("--- Processamento em Lote Concluído ---")
    print(f"Imagens processadas com sucesso: {processed_count}")
    print(f"Falhas: {failed_count}")