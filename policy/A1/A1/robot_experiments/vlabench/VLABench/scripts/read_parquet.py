import pandas as pd
import argparse
from pathlib import Path

def read_parquet(file_path: str, columns: list = None):
    """
    读取parquet文件
    
    参数:
        file_path (str): parquet文件的路径
        columns (list, optional): 要读取的列名列表。如果为None，则读取所有列
    
    返回:
        pandas.DataFrame: 读取的数据
    """
    try:
        # 检查文件是否存在
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件 {file_path} 不存在")
            
        # 检查文件扩展名
        if not file_path.endswith('.parquet'):
            raise ValueError("文件必须是.parquet格式")
            
        # 读取parquet文件
        if columns:
            df = pd.read_parquet(file_path, columns=columns)
        else:
            df = pd.read_parquet(file_path)
            
        return df
        
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='读取parquet文件')
    parser.add_argument('file_path', type=str, help='parquet文件的路径')
    parser.add_argument('--columns', type=str, nargs='+', help='要读取的列名（可选）')
    parser.add_argument('--head', type=int, default=5, help='显示前n行数据（默认5行）')
    
    args = parser.parse_args()
    
    # 读取文件
    df = read_parquet(args.file_path, args.columns)
    
    if df is not None:
        print(f"\n文件 {args.file_path} 的基本信息:")
        print(f"行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print(f"\n列名: {list(df.columns)}")
        print(f"\n前{args.head}行数据:")
        print(df.head(args.head))
        
        # 保存为CSV文件（可选）
        csv_path = Path(args.file_path).with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n数据已保存为CSV文件: {csv_path}")

if __name__ == "__main__":
    main() 