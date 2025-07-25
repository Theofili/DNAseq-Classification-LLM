## **Load BiomaRt**

-   Used hsapiens gene dataset

<!-- -->

    library("biomaRt")

    ## Warning: package 'biomaRt' was built under R version 4.4.2

    listMarts()

    ##                biomart                version
    ## 1 ENSEMBL_MART_ENSEMBL      Ensembl Genes 114
    ## 2   ENSEMBL_MART_MOUSE      Mouse strains 114
    ## 3     ENSEMBL_MART_SNP  Ensembl Variation 114
    ## 4 ENSEMBL_MART_FUNCGEN Ensembl Regulation 114

    ensembl <- useMart("ENSEMBL_MART_ENSEMBL")

    ensembl_datasets <- listDatasets(ensembl)
    head(ensembl_datasets)

    ##                        dataset                           description
    ## 1 abrachyrhynchus_gene_ensembl Pink-footed goose genes (ASM259213v1)
    ## 2     acalliptera_gene_ensembl      Eastern happy genes (fAstCal1.3)
    ## 3   acarolinensis_gene_ensembl       Green anole genes (AnoCar2.0v2)
    ## 4    acchrysaetos_gene_ensembl       Golden eagle genes (bAquChr1.2)
    ## 5    acitrinellus_gene_ensembl        Midas cichlid genes (Midas_v5)
    ## 6    amelanoleuca_gene_ensembl       Giant panda genes (ASM200744v2)
    ##       version
    ## 1 ASM259213v1
    ## 2  fAstCal1.3
    ## 3 AnoCar2.0v2
    ## 4  bAquChr1.2
    ## 5    Midas_v5
    ## 6 ASM200744v2

    ensembl <- useDataset("hsapiens_gene_ensembl", mart= ensembl)

    filters = listFilters(ensembl)
    write.table(filters, "filters.tsv", sep = '\t', row.names =
    FALSE, quote = FALSE)

    attributes = listAttributes(ensembl)
    write.table(attributes, "attributes.tsv", sep = '\t', row.names
    = FALSE, quote = FALSE)

    # Which attributes are sequences
    seqs_attributes <- attributes[which(attributes$page == 'sequences'),]

### Define attributes and filters, getBM

    attributes01 <- c(
         "ensembl_gene_id",
         "gene_biotype",
         'description',
         "coding",
         "peptide")

    filters01 <- c("chromosome_name", "biotype")

    values01 <- list(chromosome_name=1, biotype='protein_coding')

## Look though all chromosome protein coding values

    # All chromosomeses, meaning repeat 4 times with different chromosomes

    values02 <- list(chromosome_name=c('1', '2','3', '4', '5' ), biotype='protein_coding')

    info02 <- getBM(
         attributes=attributes01,
         filters=filters01,
         values=values02,
         mart=ensembl)

### Filter output

    # Drop sequence unavailable

    info02 <- info02[which(info02$coding != 'Sequence unavailable'),]

    colnames(info02) <-c('descriprion', 'peptide', 'coding', 'peptide', 'ensembl_gene_id')


    # For some reason there are empty values on this attribute
    info02$peptide <- NULL

    info_copy <- info02

## Add the type of protein corresponds to the value

    # 

    library(stringr)

    ## Warning: package 'stringr' was built under R version 4.4.2

    # Classify into membrane, nuclear(Have similar number of outputs)
    membrane <- info_copy[str_detect(info_copy$descriprion, 'membrane'),]
    nuclear <- info_copy[str_detect(info_copy$descriprion, 'nuclear'),]



    nuclear$type <- 'nuclear'
    membrane$type <- 'membrane'

    data <- rbind(nuclear, membrane)

    data_copy <- data

    data_copy$description <- NULL# AND SO ON until there is only coding and type
    data_copy$ensembl_gene_id <- NULL
    data_copy$gene_biotype <- NULL



    write.csv(data_copy, file='C:/Users/theos/Desktop/internship/protein_data.csv')
