## Write a function to retrieve words that appear often on the description column and add them into a label column

Preperation for multi-label Classification

    ## Example of labels that could be used - too many

    labels <- c('kinase', 'zinc finger', 'G protein', 'receptor', 
                'carrier', 'mitochondrial', 'binding', 'transcription',
                'membrane', 'nuclear', 'secreted', 'cytosolic', 'accessory',
                'non-receptor', 'transmembrane', 'vesicle', 'interacting', 
                'interleukin', 'suppressor', 'hormone', 'adaptor', 'lysosomal', 
                'repeat', 'leucine', 'regulatory factor', 'interferon', 'coenzyme', 
                'acyltransferase', 'dehydrogenase', 'synthetase', 'phosphodiesterase',
                'oxygenase', 'phosphatase', 'reductase', 'phosphoprotein', 'actin', 
                'transferase', 'repressor' )


    # Create a function to label data into the words that appear in the description

    get_labels <- function(df, labels) {
      
      # Column for every label type
      for (label in labels) {
        df[[label]] <- as.character(NA)
      }
      
      for (i in 1:nrow(df)){
        for (label in labels){
          
          if (str_detect(df$description[i], label)){
              
              df[[label]][i] <- label
              #break 
          
          }
        }
      }
      return(df)
    }
