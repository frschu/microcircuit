zero <- 1e-10
gauss <- function(t, mu, s) {
  n <- 1#/(s*sqrt(2*pi))
  out <- n * exp(-0.5*((t-mu)/s)^2)
  out[out<zero] <- zero
  return(out)
}
lngauss <- function(t, mu, s, st) {
  n <- 1#/(s*sqrt(2*pi))
  out <- n * exp(-0.5*((log(st*t)-mu)/s)^2)
  out[out<zero] <- zero
  return(out)
}

input <- data.frame(time =seq(0, 348, by = 1))
input$pIkk <- (lngauss(input$time,-1,.6,.05)) + (1-exp(-input$time/10))*gauss(input$time, 30,150)*.25
input$pIkk <- input$pIkk/max(input$pIkk)

g <- ggplot(input, aes(x=time, y=pIkk)) + geom_line()
g <- g + theme(panel.grid.minor = element_line(color='white'),axis.text=element_text(colour="black"))
g <- g + xlab('time [min]') + ylab('pIKK [a.u.]')
print(g)
ggsave(filename='pIkk.pdf',height=8,width = 10,units='cm')

input$epsilon <- 0

input2 <- data.frame(time =seq(0, 348, by = 1))
input2$epsilon <- -.45
input2$pIkk <- input$pIkk/(1+input2$epsilon*input$pIkk)
input2$pIkk <- input2$pIkk/max(input2$pIkk)

input3 <- data.frame(time =seq(0, 348, by = 1))
input3$epsilon <- -.9
input3$pIkk <- input$pIkk/(1+input3$epsilon*input$pIkk)
input3$pIkk <- input3$pIkk/max(input3$pIkk)

input4 <- data.frame(time =seq(0, 348, by = 1))
input4$epsilon <- 3
input4$pIkk <- input$pIkk/(1+input4$epsilon*input$pIkk)
input4$pIkk <- input4$pIkk/max(input4$pIkk)

input5 <- data.frame(time =seq(0, 348, by = 1))
input5$epsilon <- 10
input5$pIkk <- input$pIkk/(1+input5$epsilon*input$pIkk)
input5$pIkk <- input5$pIkk/max(input5$pIkk)

inputT <- rbind(input,input2,input3,input4,input5)

inputT$epsilon <- factor(inputT$epsilon)

g <- ggplot(inputT, aes(x=time, y=pIkk, group = epsilon, color = epsilon)) + geom_line() + labs(colour=expression(epsilon))
g <- g + theme(panel.grid.minor = element_line(color='white'),axis.text=element_text(colour="black"))
g <- g + xlab('time [min]') + ylab('pIKK [a.u.]')# + scale_colour_gradient(low="#353A4E", high ="#A0AFEA")
g <- g + theme(legend.position = c(0, 1),
              legend.justification = c(-2.2, .955), 
              legend.background = element_rect(colour = NA, fill = "transparent"),
               legend.key = element_rect(fill = 'white'))
print(g)

ggsave(filename='pIkk_trans.pdf',height=8,width = 10,units='cm')





input <- data.frame(time =seq(0, 348, by = 1))
input$pIkk <- (lngauss(input$time,-1,.6,.05)) + (1-exp(-input$time/10))*gauss(input$time, 30,150)*.25
input$pIkk <- input$pIkk/max(input$pIkk)

# inputM <- input
# input2 <- data.frame(time =seq(0, 348, by = 1))
# input2$epsilon <- 1
# input2$pIkk <- input$pIkk/(1+input2$epsilon*input$pIkk)
# input2$pIkk <- input2$pIkk/max(input2$pIkk)


input$modified <- (lngauss(input$time,-1,.6,.05)) + (1-exp(-input$time/10))*gauss(input$time, 30,150)*.46
input$modified <- input$modified/max(input$modified)
input$transformed <- input$modified/(1-.5*input$modifie)
input$transformed <- input$transformed/max(input$transform)
# 1 <-> .46
# 2 <-> .64

# input$modified2 <- (lngauss(input$time,-1,.6,.05)) + (1-exp(-input$time/10))*gauss(input$time, 30,150)*.05
# input$modified2 <- input$modified2/max(input$modified2)
# input$transformed2 <- input$modified2/(1+5*input$modified2)
# input$transformed2 <- input$transformed2/max(input$transformed2)

out <- melt(input, id.vars = 'time', var.name='pIkk')

g <- ggplot(out, aes(x=time, y=value, group = variable, color = variable)) + geom_line()
g <- g + theme(panel.grid.minor = element_line(color='white'),axis.text=element_text(colour="black"))
g <- g + xlab('time [min]') + ylab('pIKK [a.u.]')
g <- g + theme(legend.position='none')

g <- g + geom_segment(size = .3, aes(x = 55, y = 0, xend = 55,yend = .22), color = 'black', arrow = arrow(length = unit(2, "mm"), ends="both"))
g <- g + geom_segment(size = .3, aes(x = 70, y = 0, xend = 70,yend = .355), color = 'black', arrow = arrow(length = unit(2, "mm"), ends="both"))
g <- g + geom_segment(size = .3, aes(x = 95, y = 0.34, xend = 95,yend = .2), color = 'black', arrow = arrow(length = unit(2, "mm"), ends="last"))
g <- g + annotate("text", x = 44, y = .115, label = 'h')
g <- g + annotate("text", x = 84, y = .115, label = 'h*')
g <- g + annotate("text", x = 113, y = .26, label = 'X[epsilon]',parse=TRUE)

  # print(g)

ggsave(filename='pIkk_modified.pdf',height=8,width = 10,units='cm')
