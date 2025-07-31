# **Overview**

#### Use this pipeline to create a Dataframe of labeled sequences

-   Use the BioMart Database - *Homo sapiens* genes
-   Filter throught the sequences containing protein coding genes
-   Use the function to filter data from desired chromosomes and desired
    labels

## 1. Acess the Biomart Database, prepare variables to be searched

    # Load libraries essential for this pipeline

    library("biomaRt")

    ## Warning: package 'biomaRt' was built under R version 4.4.2

    library("stringr")

    ## Warning: package 'stringr' was built under R version 4.4.2

    # Load Homo sapiens genes dataset

    ensembl <- useMart("ENSEMBL_MART_ENSEMBL")
    ensembl <- useDataset("hsapiens_gene_ensembl", mart=ensembl)

    # Establish attributes

    attr <- c(
      "ensembl_gene_id",
      "gene_biotype",
      "description",
      "coding"
    )

    # Initialize the way of filtering 

    fil <- c("chromosome_name", "biotype")

## 2. Create a function to access specific chromosomes and labels

    # Initialize the function with 3 parameters

    get_data <- function(chr_num, label_1, label_2){
      
      # Make sure yout parameters are correct
      print(paste("chr_num:", chr_num))
      print(paste("label_1:", label_1))
      print(paste("label_2:", label_2))
      

      # Set the filter values
      
      val <- list(chromosome_name = chr_num, biotype = 'protein_coding')
      
      
      # Run through the Database to get your data into a df
      inf <- getBM(
        attributes = attr,
        filters = fil,
        values = val,
        mart = ensembl
      )
      
      # Check if there are matches to your filtered check
      
      if (nrow(inf) == 0) {
        warning("No data returned from getBM().")
        return(data.frame())
      }
      
      
      # Remove unavailable sequences
      
      inf <- inf[which(inf$coding != 'Sequence unavailable'), ]
      
      
      # Assign column names
      
      colnames(inf) <- c('description', 'coding', 'ensembl_gene_id', 'gene_biotype')
      
      # Filter by labels
      
      label_1_df <- inf[str_detect(inf$description, label_1), ]
      label_2_df <- inf[str_detect(inf$description, label_2), ]

      
      # Check that there are sequences corresponding to your labels
      
      if (nrow(label_1_df) > 0) {
        label_1_df$type <- label_1 
      }
      if (nrow(label_2_df) > 0) {
        label_2_df$type <- label_2 
      }
      
      df <- rbind(label_1_df, label_2_df)
      
      
      if (nrow(df) == 0) {
        warning("No matching genes found for the provided labels.")
        return(data.frame())
      }
      
      # Save dataframe with all columns
      
      df_all <<-as.data.frame(df)
      
      # Remove columns not needed in the classification model
      
      df$description <- NULL
      df$ensembl_gene_id <- NULL
      df$gene_biotype <- NULL
      
      # Name the output approprietly 
      
      df_name = paste(label_1, label_2, sep='_')
      
      df_name <<-as.data.frame(df)
      return(df_name)
    }

## 3. Function output

    get_data('1', 'kinse', 'receptor')

    ## [1] "chr_num: 1"
    ## [1] "label_1: kinse"
    ## [1] "label_2: receptor"

    ## [1] "kinse_receptor"

    head(df_name, 1)

    ##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             coding
    ## 20 ATGGAACCGGCCCCCTCCGCCGGCGCCGAGCTGCAGCCCCCGCTCTTCGCCAACGCCTCGGACGCCTACCCTAGCGCCTGCCCCAGCGCTGGCGCCAATGCGTCGGGGCCGCCAGGCGCGCGGAGCGCCTCGTCCCTCGCCCTGGCAATCGCCATCACCGCGCTCTACTCGGCCGTGTGCGCCGTGGGGCTGCTGGGCAACGTGCTTGTCATGTTCGGCATCGTCCGGTACACTAAGATGAAGACGGCCACCAACATCTACATCTTCAACCTGGCCTTAGCCGATGCGCTGGCCACCAGCACGCTGCCTTTCCAGAGTGCCAAGTACCTGATGGAGACGTGGCCCTTCGGCGAGCTGCTCTGCAAGGCTGTGCTCTCCATCGACTACTACAATATGTTCACCAGCATCTTCACGCTCACCATGATGAGTGTTGACCGCTACATCGCTGTCTGCCACCCTGTCAAGGCCCTGGACTTCCGCACGCCTGCCAAGGCCAAGCTGATCAACATCTGTATCTGGGTCCTGGCCTCAGGCGTTGGCGTGCCCATCATGGTCATGGCTGTGACCCGTCCCCGGGACGGGGCAGTGGTGTGCATGCTCCAGTTCCCCAGCCCCAGCTGGTACTGGGACACGGTGACCAAGATCTGCGTGTTCCTCTTCGCCTTCGTGGTGCCCATCCTCATCATCACCGTGTGCTATGGCCTCATGCTGCTGCGCCTGCGCAGTGTGCGCCTGCTGTCGGGCTCCAAGGAGAAGGACCGCAGCCTGCGGCGCATCACGCGCATGGTGCTGGTGGTTGTGGGCGCCTTCGTGGTGTGTTGGGCGCCCATCCACATCTTCGTCATCGTCTGGACGCTGGTGGACATCGACCGGCGCGACCCGCTGGTGGTGGCTGCGCTGCACCTGTGCATCGCGCTGGGCTACGCCAATAGCAGCCTCAACCCCGTGCTCTACGCTTTCCTCGACGAGAACTTCAAGCGCTGCTTCCGCCAGCTCTGCCGCAAGCCCTGCGGCCGCCCAGACCCCAGCAGCTTCAGCCGCGCCCGCGAAGCCACGGCCCGCGAGCGTGTCACCGCCTGCACCCCGTCCGATGGTCCCGGCGGTGGCGCTGCCGCCTGA
    ##        type
    ## 20 receptor
