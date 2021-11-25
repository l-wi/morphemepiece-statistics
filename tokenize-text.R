#!/usr/bin/env Rscript


to_str <- function(x) {
    return (paste(names(x), collapse=" "))
}

library(morphemepiece)

stdin_con = file("stdin")
inp <- readLines(con = stdin_con)
close(stdin_con)

res <- morphemepiece_tokenize(inp)

res <- lapply(res,to_str )

cat(unlist(res))
#cat(paste(names(res[[1]]), collapse=" "))

