import cv2
import numpy as np
import os

def apply_canny_filter(input_path: str, output_path: str):
    """
    Aplica o filtro de bordas Canny com parâmetros fixos a uma imagem
    e a salva no caminho de saída.

    Args:
        input_path (str): O caminho da imagem a ser processada.
        output_path (str): O caminho onde a imagem processada será salva.
    
    Returns:
        bool: True se o processo foi bem-sucedido, False caso contrário.
    """
    try:
        # Define os parâmetros perfeitos que você encontrou no calibrador
        blur_ksize = 9
        threshold1 = 0
        threshold2 = 20

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

        # 3. Aplica o detector de bordas Canny com os thresholds definidos
        canny_edges = cv2.Canny(blurred_image, threshold1, threshold2)
        
        # Salva a imagem resultante (o resultado do Canny já é uma imagem P&B)
        cv2.imwrite(output_path, canny_edges)
        return True

    except Exception as e:
        print(f"  [ERRO] Ocorreu uma exceção ao processar '{input_path}': {e}")
        return False


def main():
    """
    Função principal que varre os diretórios, aplica o filtro Canny
    e salva os resultados, replicando a estrutura de pastas.
    """
    root_input_dir = "images/crop_only"
    root_output_dir = "images/crop_and_canny"

    valid_extensions = ('.png')

    print(f"--- Iniciando Processamento em Lote (Filtro Canny) ---")
    print(f"Diretório de Origem: {root_input_dir}")
    print(f"Diretório de Saída:  {root_output_dir}")
    print(f"Parâmetros: blur_ksize={9}, threshold1={0}, threshold2={20}")
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
                
                # Chama a função para aplicar o filtro Canny
                success = apply_canny_filter(input_path, output_path)
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