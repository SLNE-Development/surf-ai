# Note: this project's Kotlin toolchain compiles to Java 25 bytecode (verified via
# `javap -verbose`, major version 69) - eclipse-temurin:21 would refuse to run it.
FROM gradle:9-jdk25 AS build
WORKDIR /workspace
COPY . .
RUN gradle :surf-ai-microservice:shadowJar --no-daemon

FROM eclipse-temurin:25-jre
WORKDIR /app
COPY --from=build /workspace/surf-ai-microservice/build/libs/*-all.jar /app/app.jar
VOLUME ["/app/config"]
CMD ["java", "-jar", "/app/app.jar"]
