import pandas as pd

# 1. Carregar os dados do ficheiro CSV
# (Certifica-te que o ficheiro se chama 'dados.csv' e está na mesma pasta)
df = pd.read_csv('Student_performance_data.csv')

# 2. Inverter a coluna GradeClass
# A fórmula matemática (4.0 - x) faz exatamente a inversão pedida:
# 4.0 - 0.0 = 4.0
# 4.0 - 1.0 = 3.0
# 4.0 - 2.0 = 2.0
# 4.0 - 3.0 = 1.0
# 4.0 - 4.0 = 0.0
df['GradeClass'] = 4.0 - df['GradeClass']

# 3. (Opcional) Converter para inteiro caso queiras os números sem casa decimal (ex: 5 em vez de 5.0)
# df['GradeClass'] = df['GradeClass'].astype(int)

# 4. Guardar o resultado num novo ficheiro CSV
df.to_csv('dados_grade_invertida.csv', index=False)

print("A coluna GradeClass foi invertida com sucesso e guardada em 'dados_grade_invertida.csv'!")