import ocr_pipeline
import onet_classification

# Main Function
def main():
    ocr_pipeline.main()
    matches = onet_classification.main()

    ocr = pd.read_csv('D:/nlp/top_jobs_cs_20_21/part_1/part_1b/p1b.csv')
    
    return (ocr, matches)

if __name__ == '__main__':
    main()