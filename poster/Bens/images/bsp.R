library(deSolve)
library(reshape2)
library(ggplot2)
  
  LVmod <- function(Time, State, Pars) {
  with(as.list(c(State, Pars)), {
    
    dA <- -2*k1*A*A + 2*k2*B
    dB <- k1*A*A - k2*B
    
    return(list(c(dA, dB)))
  })
}

pars  <- c(k1   = .1,
           k2  = .5)

yini  <- c(A = 1, B = 0)
times <- seq(0, 10, by = .1)
out   <- ode(yini, times, LVmod, pars)
out <- as.data.frame(out)

out$C <- 10 *out$A
out$scale = 1

outS <- out

for (i in c(2,3,4)){
  
  frame <- data.frame(time = seq(0,10,by=.1), A = i*outS$A, B = i*outS$B, C = outS$C, scale = i)
  
  out <- rbind(out,frame)
}
colnames(out)[[4]] <- "Aobs"
out <- melt(out,id.vars=c("time","scale"), measure.vars=c("A","B","Aobs"))

#out$scale <- factor(out$scale)



p <- ggplot(out, aes(x=time, y=value, group=scale, color=scale)) + geom_line() + facet_wrap(~variable, scales="free")
p <- p + scale_colour_gradient(low="gray10", high ="#7782AF") + labs(colour=expression(phantom(x)*e^epsilon))
p <- p + xlab('time') + ylab('value')
print(p)

ggsave(filename='bsp.pdf',height= 6,width = 20,units='cm')
