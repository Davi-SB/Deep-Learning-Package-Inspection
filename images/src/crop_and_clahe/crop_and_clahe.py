import cv2
import numpy as np
import os

def apply_clahe_filter(input_path: str, output_path: str):
    try:
        # Define os parâmetros perfeitos que você encontrou no calibrador
        clip_limit = 10.0
        tile_size = 5

        # Lê a imagem de entrada
        image = cv2.imread(input_path)
        if image is None:
            print(f"  [ERRO] Não foi possível ler o arquivo: {input_path}")
            return False

        # --- Aplica o pipeline do filtro CLAHE ---
        # 1. Converte a imagem para o espaço de cor LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 2. Separa os canais L, a, b
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # 3. Cria o objeto CLAHE com os parâmetros definidos
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        # 4. Aplica o CLAHE APENAS no canal L (Luminosidade)
        l_clahe = clahe.apply(l_channel)

        # 5. Junta o novo canal L com os canais a e b originais
        lab_clahe_image = cv2.merge((l_clahe, a_channel, b_channel))

        # 6. Converte a imagem de volta para o espaço BGR
        final_image = cv2.cvtColor(lab_clahe_image, cv2.COLOR_LAB2BGR)
        
        # Salva a imagem resultante
        cv2.imwrite(output_path, final_image)
        return True

    except Exception as e:
        print(f"  [ERRO] Ocorreu uma exceção ao processar '{input_path}': {e}")
        return False


def main():
    root_input_dir = "images/crop_only"
    root_output_dir = "images/crop_and_clahe"

    valid_extensions = ('.png')

    print(f"--- Iniciando Processamento em Lote (Filtro CLAHE) ---")
    print(f"Diretório de Origem: {root_input_dir}")
    print(f"Diretório de Saída:  {root_output_dir}")
    print(f"Parâmetros: clipLimit=10.0, tileGridSize=(5, 5)")
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
                
                # Chama a função para aplicar o filtro CLAHE
                success = apply_clahe_filter(input_path, output_path)
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