plugins {
	java
	id("org.springframework.boot") version "3.3.3"
	id("io.spring.dependency-management") version "1.1.6"
}

group = "com"
version = "0.0.1-SNAPSHOT"

java {
	toolchain {
		languageVersion = JavaLanguageVersion.of(21)
	}
}

repositories {
	mavenCentral()
}

dependencies {
	implementation("org.springframework.boot:spring-boot-starter-web")
	testImplementation("org.springframework.boot:spring-boot-starter-test")
	testRuntimeOnly("org.junit.platform:junit-platform-launcher")
	implementation("net.sourceforge.tess4j:tess4j:5.13.0")
	implementation("org.openpnp:opencv:4.9.0-0")
	implementation("com.microsoft.onnxruntime:onnxruntime:1.19.2")
	implementation("ai.djl:api:0.30.0")
	implementation("ai.djl.pytorch:pytorch-engine:0.30.0")
	runtimeOnly("ai.djl.mxnet:mxnet-engine:0.30.0")
	implementation("ai.djl:model-zoo:0.30.0")
	implementation("tech.tablesaw:tablesaw-core:0.43.1")
	implementation("com.googlecode.json-simple:json-simple:1.1")
}

tasks.withType<Test> {
	useJUnitPlatform()
}
