# save out VP misslist
save_VPmisslist<-function(virtual,hivaefolder){
  
  # for every zcode variable (autoencoder var group) in the data
  for (code in colnames(virtual)[grepl('zcode',colnames(virtual))]){
    
    group<-gsub('zcode_','',code)
    visits<-c('_VIS00','_VIS01','_VIS02','_VIS03','_VIS04','_VIS05','_VIS06','_VIS07','_VIS08','_VIS09','_VIS10','_VIS11','_VIS12','_VIS13','_VIS14','_VIS15')
    for (visit in visits){
      aux<-paste0('AUX_',group,visit)
      vismiss<-paste0('visitmiss',visit)
      aux_exists<-aux %in% colnames(virtual)
      vismiss_exists<-vismiss %in% colnames(virtual)

      raw<-read.csv(paste0(hivaefolder,'data_python/',group,visit,'.csv'),header=F)

      if (aux_exists | vismiss_exists){
        if (aux_exists){
          matrix<-matrix(ifelse(virtual[,aux]==1,0,1),dim(raw)[1],dim(raw)[2])
        }else{
          matrix<-matrix(ifelse(virtual[,vismiss]==1,0,1),dim(raw)[1],dim(raw)[2])
        }
      }else{
        matrix<-matrix(1,dim(raw)[1],dim(raw)[2])
      }
      # write to file in the hivae folder
      write.table(matrix,paste0(hivaefolder,'VP_misslist/',group,visit,'_vpmiss.csv'), sep=",",row.names = F,col.names = F)
    }
  }
  
}