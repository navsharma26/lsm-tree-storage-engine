# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod ./
COPY pkg/ ./pkg/
COPY cmd/ ./cmd/
COPY web/ ./web/

RUN CGO_ENABLED=0 GOOS=linux go build -o lsm-server ./cmd/server/main.go

# Production runner stage
FROM alpine:3.19
RUN apk --no-cache add ca-certificates

WORKDIR /app
COPY --from=builder /app/lsm-server .
COPY web/ ./web/

ENV PORT=8080
EXPOSE 8080

CMD ["./lsm-server", "-port", "8080", "-dir", "./data_cloud"]
