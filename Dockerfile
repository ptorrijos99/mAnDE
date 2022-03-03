#
# Build stage
#
FROM maven:3.6.3-jdk-8-slim AS build
MAINTAINER Pablo Torrijos Arenas <Pablo.Torrijos@uclm.es>
COPY src /src
COPY repos /repos
COPY pom.xml /
RUN mvn -f /pom.xml clean package


#
# Package stage
#
FROM openjdk:8-jre-slim
COPY --from=build /target/mAnDE-3.0-jar-with-dependencies.jar /experiment.jar
ENTRYPOINT ["java","-jar","/experiment.jar"]