
library(shiny)
shinyUI(fluidPage(
  sidebarLayout(
    sidebarPanel(
      fileInput("file1", "Choose CSV File",
                accept = c(
                  "text/csv",
                  "text/comma-separated-values,text/plain",
                  ".csv")
      ),
      tags$hr(),
      checkboxInput("header", "Header", TRUE),
      fileInput("file2", "Choose CSV File",
                accept = c(
                  "text/csv",
                  "text/comma-separated-values,text/plain",
                  ".csv")
      ),
      tags$hr(),
      checkboxInput("header", "Header", TRUE)
    ),
    mainPanel(
      titlePanel("Bagging:\n"),
      tableOutput("bagging"),
      titlePanel("Boosting"),
      tableOutput("boosting"),
      titlePanel("Support Vector Regression"),
      tableOutput("svr"),
      titlePanel("Random Forest"),
      tableOutput("rforest")
    )
  )
  
  
)

)
