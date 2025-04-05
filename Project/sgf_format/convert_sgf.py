import os

def convert_to_sgf(movements_file: str, output_file: str):
    sgf_header = "(;GM[Amazons]SZ[8]"
    sgf_moves = []

    def square_to_sgf(sq):
        return f"{sq[0]}{sq[1]}"

    with open(movements_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split("-")
            if len(parts) < 2:
                continue
            move_info = parts[1].strip()

            if move_info.startswith("W:"):
                player = "W"
                move_str = move_info[2:].strip()
            elif move_info.startswith("B:"):
                player = "B"
                move_str = move_info[2:].strip()
            else:
                continue

            splitted = move_str.split(",")
            first_part = splitted[0].strip()
            arrow_part = splitted[1].strip() if len(splitted) > 1 else None

            from_to = first_part.split("->")
            if len(from_to) == 2:
                from_square = from_to[0].strip()
                to_square = from_to[1].strip()
                sgf_moves.append(f";{player}[{from_square}{to_square}]")

            if arrow_part and arrow_part.startswith("A:"):
                arrow_square = arrow_part[2:].strip()
                sgf_moves.append(f";AE[{arrow_square}]")

    sgf_content = sgf_header + "".join(sgf_moves) + ")"
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(sgf_content)

    print(f"Archivo SGF generado en: {output_file}")

if __name__ == "__main__":
    input_path = r"C:\Users\johan\OneDrive\Escritorio\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\Project\sgf_format\movimientos.txt"
    output_path = r"C:\Users\johan\OneDrive\Escritorio\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\Project\sgf_format\partida.sgf"
    convert_to_sgf(input_path, output_path)
