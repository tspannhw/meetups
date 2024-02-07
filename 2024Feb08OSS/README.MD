

# Flink Tables

````
CREATE TABLE `ssb`.`Meetups`.`jsontranscom` (
  `title` VARCHAR(2147483647),
  `description` VARCHAR(2147483647),
  `pubDate` VARCHAR(2147483647),
  `point` VARCHAR(2147483647),
  `latitude` VARCHAR(2147483647),
  `ts` VARCHAR(2147483647),
  `longitude` VARCHAR(2147483647),
  `uuid` VARCHAR(2147483647),
  `eventTimeStamp` TIMESTAMP(3) WITH LOCAL TIME ZONE METADATA FROM 'timestamp',
  WATERMARK FOR `eventTimeStamp` AS `eventTimeStamp` - INTERVAL '3' SECOND
) WITH (
  'scan.startup.mode' = 'earliest-offset',
  'deserialization.failure.policy' = 'ignore_and_log',
  'properties.request.timeout.ms' = '120000',
  'properties.auto.offset.reset' = 'earliest',
  'format' = 'json',
  'properties.bootstrap.servers' = 'kafka:9092',
  'connector' = 'kafka',
  'properties.transaction.timeout.ms' = '900000',
  'topic' = 'transcom'
)
````
